from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from modules.pipeline.linking_metadata import AddressLinkingMetadata

class Storage(ABC):
    @abstractmethod
    def upsert(self, linked_address : 'AddressLinkingMetadata') -> None:
        """
        Update or insert the linked address in the storage after applying a linking step
        """
        pass

    @abstractmethod
    def fetch_pending(self, n : int = 1) -> Optional['AddressLinkingMetadata'] | list['AddressLinkingMetadata']:
        """
        Fetch pending addresses for processing, e.g. for applying a linking step
        """
        pass

    @abstractmethod
    def get_pending_count(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        pass

    @abstractmethod
    def get_total_count(self) -> int:
        """
        Get the total number of addresses in the storage, e.g. for logging or monitoring purposes
        """
        pass
