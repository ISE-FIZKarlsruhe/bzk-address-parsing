from modules.pipeline.storage.storage import Storage
from typing import Optional
from modules.pipeline.linking_metadata import AddressLinkingMetadata
from pathlib import Path
import json
from modules.pipeline.storage.encoding_util import encode_as_dict, decode_from_dict
import asyncio
import time
from copy import deepcopy

class DeferredJsonStorage(Storage):
    def __init__(self, filepath : str | Path, defer_write_seconds : float = 5.0):
        self.filepath = Path(filepath)
        self.memory = dict()
        self.pending : set[str] = set()
        self.defer_write_seconds = defer_write_seconds
        self._deferred_write_task : Optional[asyncio.Task] = None
        self._write_time : Optional[float] = None
    


    def initialize(self):
        if self.filepath.exists():
            with self.filepath.open("r", encoding="utf-8") as f:
                self.memory = json.load(f)
            for address_id, data in self.memory.items():
                if not data["finished"]:
                    self.pending.add(address_id)

    async def _wait_then_write(self):
        now = time.monotonic()
        if now < self._write_time:
            await asyncio.sleep(self._write_time - now)
        with self.filepath.open("w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)
        self._deferred_write_task = None
        self._write_time = None

    async def _defer_write(self):
        now = time.monotonic()
        if self._write_time is not None:
            if now >= self._write_time:
                # If the scheduled write time has already passed, write immediately
                with self.filepath.open("w", encoding="utf-8") as f:
                    json.dump(self.memory, f, ensure_ascii=False, indent=4)
                if self._deferred_write_task is not None:
                    self._deferred_write_task.cancel()
                    self._deferred_write_task = None
                self._write_time = None
        else:
            self._write_time = now + self.defer_write_seconds
            self._deferred_write_task = asyncio.create_task(
                self._wait_then_write(), name=f"{self.__class__.__name__}-DeferredWrite")

    async def upsert(self, linked_address : AddressLinkingMetadata, preserve_linking_data : bool = False) -> None:
        """
        Update or insert the linked address in the storage after applying a linking step
        """
        new_data = deepcopy(encode_as_dict(linked_address))
        if preserve_linking_data and linked_address.address.id in self.memory:
            existing_data = self.memory[linked_address.address.id]["address"]
            new_data["address"] = existing_data
        self.memory[linked_address.address.id] = new_data
        if not linked_address.finished:
            self.pending.add(linked_address.address.id)
        await self._defer_write()

    
    async def fetch_pending(self, n : int = 1) -> Optional[AddressLinkingMetadata] | list[AddressLinkingMetadata]:
        """
        Fetch pending addresses for processing, e.g. for applying a linking step
        """
        id = self.pending.pop() if self.pending else None
        if id is None:
            return None if n == 1 else []
        data = self.memory[id]
        linked_address : AddressLinkingMetadata = decode_from_dict(data, AddressLinkingMetadata)
        return linked_address if n == 1 else [linked_address]

    async def  get_pending_count(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        return len(self.pending)

    async def  get_total_count(self) -> int:
        """
        Get the total number of addresses in the storage, e.g. for logging or monitoring purposes
        """
        return len(self.memory)

    async def finalize(self) -> None:
        """
        Close the storage and release any resources, e.g. file handles or database connections
        """
        if self._deferred_write_task is not None:
            self._deferred_write_task.cancel()
            self._deferred_write_task = None
        with self.filepath.open("w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)
        pass