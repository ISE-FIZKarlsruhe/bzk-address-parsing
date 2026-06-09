
from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult
from modules.pipeline.linking_steps import LinkingStepResult
from modules.pipeline.executors.executor_context import registered_multiprocessing_resources
from modules.pipeline.executors.step_executor import StepExecutor
from modules.pipeline.executors.executor_context import ExecutorContext
import asyncio
import time

def _worker_process_apply(step_id : int, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
    global registered_multiprocessing_resources
    step = registered_multiprocessing_resources[step_id]
    return step.apply(address)

class MultiprocessingStepExecutor(StepExecutor):
    def __init__(self, step):
        self.step = step
        super().__init__()
        self._pending_count = 0
        self.rate = 0.0
    
    def initialize(self, executor_context : ExecutorContext):
        super().initialize(executor_context)
        executor_context.register_multiprocessing_resource(self.step)
        pass

    def get_pending_count(self):
        return self._pending_count

    def get_rate(self) -> float:
        return self.rate

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        self._pending_count += 1
        step_id = id(self.step)
        pool = self.context.multiprocessing_pool
        assert pool is not None, "Multiprocessing pool is not initialized"
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        resulting_data, result = await loop.run_in_executor(pool, _worker_process_apply, step_id, address)
        elapsed = time.monotonic() - start
        self.rate = 1 / elapsed if elapsed > 0 else float("inf")
        self._pending_count -= 1
        return resulting_data, result

    def finalize(self):
        # Finalize is handled by killing the multiprocessing pool
        pass