from abc import ABC, abstractmethod
from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.linked_data import AddressProcessingData
from modules.pipeline.linking_steps import LinkingStepResult
from typing import Optional

class StepExecutor(ABC):
    def __init__(self):
        self.context : Optional[ExecutorContext] = None

    @abstractmethod
    def get_pending_count(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        pass

    @abstractmethod
    def get_rate(self) -> float:
        """
        Get the estimated rate of processing for this step, e.g. for logging or monitoring purposes
        In most situations this is estimated only from the time elapsed for the last operation.
        """
        pass

    def initialize(self, executor_context : ExecutorContext):
        self.executor_context = executor_context

    @abstractmethod
    async def apply(self, address : AddressProcessingData) -> tuple[AddressProcessingData, LinkingStepResult]:
        pass
    
    @abstractmethod
    def finalize(self):
        pass