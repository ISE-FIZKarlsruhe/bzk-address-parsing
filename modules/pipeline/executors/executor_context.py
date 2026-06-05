from modules.pipeline.executors.step_executor import _initialize_worker_process
from modules.pipeline.linking_steps import LinkingStep


import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Optional

# Only set in the worker process, not in the main process
registered_multiprocessing_resources = None

def _initialize_worker_process(*, registerd_multiprocessing_steps : dict[int, LinkingStep]):
    global registered_multiprocessing_resources
    registered_multiprocessing_resources = registerd_multiprocessing_steps
    for step in registered_multiprocessing_resources.values():
        step.initialize()

class ExecutorContext:
    def __init__(
            self,
            num_workers : int | Literal['auto'] = 'auto'
        ):
        if num_workers == 'auto':
            try:
                num_workers = multiprocessing.cpu_count()
            except NotImplementedError:
                num_workers = None

        self.num_workers : int = num_workers or 8
        self.registered_multiprocessing_steps : dict[int, LinkingStep] = dict()
        self.multiprocessing_pool : Optional[ProcessPoolExecutor] = None

    def initialize(self):
        self.multiprocessing_pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_initialize_worker_process,
            initargs=(self.registered_multiprocessing_steps,)
        )

    def register_multiprocessing_resource(self, resource):
        resource_id = id(resource)
        self.registered_multiprocessing_steps[resource_id] = resource

    def close(self):
        self.multiprocessing_pool.close()
        self.multiprocessing_pool.join()