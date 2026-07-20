from modules.pipeline.linked_data import AddressProcessingData
from modules.pipeline.linking_steps import LinkingStepResult
from modules.pipeline.executors.step_executor import StepExecutor
import time

class RoundRobinExecutor(StepExecutor):
    """
    Utility executor that wraps multiple executors and distributes the load among them in a round-robin fashion.
    """
    def __init__(self, executors : list[StepExecutor]):
        self.executors = executors
        self._next_executor_index = 0
        self.rate = 0.0
    
    def get_pending_count(self):
        return sum(executor.get_pending_count() for executor in self.executors)

    def get_rate(self):
        return self.rate

    def initialize(self, executor_context):
        super().initialize(executor_context)
        for executor in self.executors:
            executor.initialize(executor_context)
    
    
    async def apply(self, address : AddressProcessingData) -> tuple[AddressProcessingData, LinkingStepResult]:
        executor = self.executors[self._next_executor_index]
        self._next_executor_index = (self._next_executor_index + 1) % len(self.executors)
        start = time.monotonic()
        result = await executor.apply(address)
        elapsed = time.monotonic() - start
        self.rate = 1 / elapsed if elapsed > 0 else float("inf")
        return result

    def finalize(self):
        for executor in self.executors:
            executor.finalize()

    