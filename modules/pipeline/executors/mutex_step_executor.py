from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult, LinkingStep
from modules.pipeline.executors.step_executor import StepExecutor
from concurrent.futures import ThreadPoolExecutor
from modules.pipeline.executors.executor_context import ExecutorContext
import asyncio
import time
import atexit

def _init_executor_thread(step : LinkingStep):
    # This is needed to ensure that the step is initialized in the thread that will run the tasks, to avoid issues with thread affinity in some libraries (e.g. spacy)
    step.initialize()


class MutexStepExecutor(StepExecutor):
    def __init__(self, step : LinkingStep):
        self.step = step
        self.rate = 0.0
        super().__init__()
        self._thread_executor = ThreadPoolExecutor(
            max_workers=1, 
            thread_name_prefix=f"MutexStepExecutor-{step.name}"
        )
        self._pending_count = 0
    
    def initialize(self, executor_context : ExecutorContext):
        super().initialize(executor_context)
        self._thread_executor.submit(self.step.initialize).result()

    def get_pending_count(self) -> int:
        return len(self._pending_count)
    
    def get_rate(self) -> float:
        return self.rate

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        loop = asyncio.get_running_loop()
        self._pending_count += 1
        start = time.monotonic()
        resulting_data, result = await loop.run_in_executor(self._thread_executor, self.step.apply(address))
        elapsed_time = time.monotonic() - start
        self.rate = 1 / elapsed_time if elapsed_time > 0 else float("inf")
        self._pending_count -= 1
        return resulting_data, result

    def finalize(self):
        self._thread_executor.submit(self.step.finalize).result()