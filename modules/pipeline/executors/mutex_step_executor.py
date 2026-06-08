from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult, LinkingStep
from modules.pipeline.executors.step_executor import StepExecutor
import asyncio
import time

class MutexStepExecutor(StepExecutor):
    def __init__(self, step : LinkingStep, context):
        self.step = step
        self.rate = 0.0
        super().__init__(context)
        self.lock = asyncio.Lock()
    
    def initialize(self):
        self.step.initialize()

    def get_pending_count(self) -> int:
        return len(self.lock._waiters)
    
    def get_rate(self) -> float:
        return self.rate

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        async with self.lock:
            start = time.monotonic()
            resulting_data, result = self.step.apply(address)
            elapsed_time = time.monotonic() - start
            self.rate = 1 / elapsed_time if elapsed_time > 0 else float("inf")
        return resulting_data, result

    def finalize(self):
        self.step.finalize()