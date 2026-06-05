from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult, LinkingStep
from modules.pipeline.executors.step_executor import StepExecutor
import asyncio

class MutexStepExecutor(StepExecutor):
    def __init__(self, step : LinkingStep, context):
        self.step = step
        super().__init__(context)
        self.lock = asyncio.Lock()
    
    def initialize(self):
        self.step.initialize()

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        async with self.lock:
            resulting_data, result = self.step.apply(address)
        return resulting_data, result

    def finalize(self):
        self.step.finalize()