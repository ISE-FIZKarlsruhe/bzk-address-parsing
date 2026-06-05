from abc import ABC, abstractmethod
from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult

class StepExecutor(ABC):
    def __init__(self, context : ExecutorContext):
        self.context = context

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        pass
    
    @abstractmethod
    def finalize(self):
        pass