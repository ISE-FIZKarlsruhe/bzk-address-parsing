from abc import ABC, abstractmethod
from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult

class StepExecutor(ABC):
    def __init__(self, context : ExecutorContext):
        self.context = context

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

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        pass
    
    @abstractmethod
    def finalize(self):
        pass