from modules.pipeline.executors.step_executor import _initialize_worker_process
from modules.pipeline.linking_steps import LinkingStep


import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Optional
import atexit

# Only set in the worker process, not in the main process
registered_multiprocessing_resources = None

def _release_worker_resources():
    global registered_multiprocessing_resources
    if registered_multiprocessing_resources is not None:
        for resource in registered_multiprocessing_resources.values():
            if hasattr(resource, "finalize") and callable(resource.finalize):
                resource.finalize()

def _initialize_worker_process(*, resgistered_multiprocessing_steps : dict[int, LinkingStep]):
    global registered_multiprocessing_resources
    registered_multiprocessing_resources = resgistered_multiprocessing_steps
    for step in registered_multiprocessing_resources.values():
        step.initialize()
    atexit.register(_release_worker_resources)

class ExecutorContext:
    def __init__(
            self,
            num_processes : int | Literal['auto'] = 'auto',
            max_concurrent_tasks : int = 100
        ):
        if num_processes == 'auto':
            try:
                num_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                num_processes = None

        self.num_processes : int = num_processes or 8
        self.max_concurrent_tasks : int = max_concurrent_tasks
        self.registered_multiprocessing_steps : dict[int, LinkingStep] = dict()
        self.multiprocessing_pool : Optional[ProcessPoolExecutor] = None

    def initialize(self):
        self.multiprocessing_pool = ProcessPoolExecutor(
            max_workers=self.num_processes,
            initializer=_initialize_worker_process,
            initargs=(self.registered_multiprocessing_steps,)
        )

    def register_multiprocessing_resource(self, resource):
        resource_id = id(resource)
        self.registered_multiprocessing_steps[resource_id] = resource

    def finalize(self):
        if self.multiprocessing_pool is not None:
            self.multiprocessing_pool.shutdown(wait=True)