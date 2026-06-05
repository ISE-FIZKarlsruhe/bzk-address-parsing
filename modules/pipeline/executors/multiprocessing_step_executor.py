
from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult
from modules.pipeline.linking_steps import LinkingStepResult
from modules.pipeline.executors.executor_context import registered_multiprocessing_resources
from modules.pipeline.step_executors import StepExecutor

def _worker_process_apply(step_id : int, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
    global registered_multiprocessing_resources
    step = registered_multiprocessing_resources[step_id]
    return step.apply(address)

class MultiprocessingStepExecutor(StepExecutor):
    def __init__(self, step, context):
        self.step = step
        super().__init__(context)
        self.context.register_multiprocessing_resource(self.step)
    
    def initialize(self):
        pass

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        step_id = id(self.step)
        pool = self.context.multiprocessing_pool
        assert pool is not None, "Multiprocessing pool is not initialized"
        resulting_data, result = await pool.submit(_worker_process_apply, step_id, address)
        return resulting_data, result

    def finalize(self):
        pass