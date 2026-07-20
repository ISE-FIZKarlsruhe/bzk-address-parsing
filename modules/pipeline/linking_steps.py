from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing import Optional
from modules.pipeline.linked_data import AddressProcessingData
from datetime import datetime


class LinkingStepResult(ABC):
    descriptive_code : str
    metadata : dict

    def __dict_encode__(self, default_encoder) -> dict:
        return default_encoder(self).update({
            "status" : self.__class__.__name__
        })

    def __dict_decode__(cls, data : dict, targs, default_decoder) -> 'LinkingStepResult':
        status = data.pop("status")
        if status == "Success":
            return default_decoder(data, Success)
        elif status == "Unresolved":
            return default_decoder(data, Unresolved)
        elif status == "Failed":
            return default_decoder(data, Failed)
        else:
            raise ValueError(f"Unknown LinkingStepResult status: {status}")

@dataclass(frozen=True)
class Success(LinkingStepResult):
    descriptive_code : str
    metadata : dict

@dataclass(frozen=True)
class Unresolved(LinkingStepResult):
    descriptive_code : str
    metadata : dict

@dataclass(frozen=True)
class Failed(LinkingStepResult):
    error_message : str
    exception_class : str
    stack_trace : str
    metadata : dict
    descriptive_code : str = "failed"

class ParallelizationType(str, Enum):
    # The process cannot be further parallelized at all
    # e.g. because it relies on a shared resource that requires mutual exclusion
    # or because it is already applying parellization internally (e.g. duckdb)
    NONE = "none"
    # The process can be parallelized for each individual item
    SIMPLE = "simple"
    # The process can be parallelized for batches of up to batch_size items
    # e.g. because it relies on a shared resource that requires mutual exclusion,
    #  but can handle batches of multiple items at once
    BATCH = "batch"

class LinkingStep(ABC):
    name : str
    parallelization_type : ParallelizationType = ParallelizationType.NONE

    def initialize(self):
        """
        Called before processing start. Can be used to allocate needed resources.
        It is called directly in the thread it will run on
        """
        pass

    @abstractmethod
    def apply(self, address : AddressProcessingData) -> tuple[AddressProcessingData, LinkingStepResult]:
        pass

    def finalize(self):
        """
        Called after processing is complete. Can be used to release allocated resources.
        It is called directly in the thread it has run on
        """
        pass

class BatchLinkingStep(LinkingStep):
    parallelization_type = ParallelizationType.BATCH
    batch_gathering_patience : Optional[float] = None
    batch_size : int

    def apply(self, address):
        return self.batch_apply([address])[0]

    @abstractmethod
    def batch_apply(self, addresses : list[AddressProcessingData]) -> list[tuple[AddressProcessingData, LinkingStepResult]]:
        pass

class CardLinkingStep(LinkingStep):
    #TODO future work: 
    # some further linking and disambiguation might be possible for complicated cases by looking at
    # other information on the card, e.g. other addresses, or the name of the card
    pass

