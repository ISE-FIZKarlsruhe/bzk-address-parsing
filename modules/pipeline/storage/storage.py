from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from modules.pipeline.linking_metadata import AddressLinkingMetadata

from concurrent.futures import ThreadPoolExecutor
import asyncio

class Storage(ABC):
    def initialize(self):
        """
        Initialize the storage, e.g. create tables or indexes if needed
        """
        pass

    @abstractmethod
    async def upsert(self, linked_address : 'AddressLinkingMetadata', preserve_linking_data : bool = False) -> None:
        """
        Update or insert the linked address in the storage after applying a linking step
        """
        pass

    @abstractmethod
    async def fetch_pending(self, n : int = 1) -> Optional['AddressLinkingMetadata'] | list['AddressLinkingMetadata']:
        """
        Fetch pending addresses for processing, e.g. for applying a linking step
        """
        pass

    @abstractmethod
    async def get_pending_count(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        pass

    @abstractmethod
    async def get_total_count(self) -> int:
        """
        Get the total number of addresses in the storage, e.g. for logging or monitoring purposes
        """
        pass

    @abstractmethod
    async def finalize(self) -> None:
        """
        Close the storage and release any resources, e.g. file handles or database connections
        """
        pass

class SynchronousStorage(ABC, Storage):
    def __init__(self):
        super().__init__()
        self._executor : Optional[ThreadPoolExecutor] = None

    @abstractmethod
    def upsert_sync(self, linked_address : 'AddressLinkingMetadata', preserve_linking_data : bool = False) -> None:
        """
        Update or insert the linked address in the storage after applying a linking step
        """
        pass

    @abstractmethod
    def fetch_pending_sync(self, n : int = 1) -> Optional['AddressLinkingMetadata'] | list['AddressLinkingMetadata']:
        """
        Fetch pending addresses for processing, e.g. for applying a linking step
        """
        pass

    @abstractmethod
    def get_pending_count_sync(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        pass

    @abstractmethod
    def get_total_count_sync(self) -> int:
        """
        Get the total number of addresses in the storage, e.g. for logging or monitoring purposes
        """
        pass

    def initialize(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{self.__class__.__name__}-Executor")
        return super().initialize()

    async def finalize(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        return super().finalize()

    def upsert(self, linked_address : 'AddressLinkingMetadata', preserve_linking_data : bool = False) -> None:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, self.upsert_sync, linked_address, preserve_linking_data)
    
    def fetch_pending(self, n : int = 1) -> Optional['AddressLinkingMetadata'] | list['AddressLinkingMetadata']:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, self.fetch_pending_sync, n)

    def get_pending_count(self) -> int:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, self.get_pending_count_sync)

    def get_total_count(self) -> int:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, self.get_total_count_sync)
    