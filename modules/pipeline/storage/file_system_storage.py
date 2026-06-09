from modules.pipeline.storage.storage import Storage
from typing import Optional
from modules.pipeline.linking_metadata import AddressLinkingMetadata
from pathlib import Path
import json
from modules.pipeline.storage.encoding_util import encode_as_dict, decode_from_dict
import asyncio

#TODO delete?

class FileSystemStorage(Storage):
    def __init__(self, directory : str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._pending_lock = asyncio.Lock()
        self.pending : set[str] = set()
    
    async def _discover_pending(self):
        for file in self.directory.glob("*.json"):
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            linking_metadata : AddressLinkingMetadata = decode_from_dict(data, AddressLinkingMetadata)
            if not linking_metadata.finished:
                async with self._pending_lock:
                    self.pending.add(file)

    async def upsert(self, linked_address : AddressLinkingMetadata) -> None:
        """
        Update or insert the linked address in the storage after applying a linking step
        """
        filepath = self.directory / f"{linked_address.address.id}.json"
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(encode_as_dict(linked_address), f, ensure_ascii=False, indent=4)
        if not linked_address.finished:
            async with self._pending_lock:
                self.pending.add(filepath)

    def fetch_pending(self, n : int = 1) -> Optional[AddressLinkingMetadata] | list[AddressLinkingMetadata]:
        """
        Fetch pending addresses for processing, e.g. for applying a linking step
        """
        fetched = []
        for _ in range(n):
            if not self.pending:
                break
            with self._pending_lock:
                filepath = self.pending.pop()
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            linked_address : AddressLinkingMetadata = decode_from_dict(data, AddressLinkingMetadata)
            fetched.append(linked_address)
        if len(fetched) == 0:
            return [] if n > 1 else None
        else:
            return fetched if n > 1 else fetched[0]

    def get_pending_count(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        return len(self.pending)

    def get_total_count(self) -> int:
        """
        Get the total number of addresses in the storage, e.g. for logging or monitoring purposes
        """
        return len(list(self.directory.glob("*.json")))
    
    def close(self) -> None:
        """
        Close the storage and release any resources, e.g. file handles or database connections
        """
        pass