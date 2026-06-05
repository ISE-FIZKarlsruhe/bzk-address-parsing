from abc import ABC, abstractmethod
from typing import Literal
from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import (
    BatchLinkingStep, ParallelizationType, LinkingStep, LinkingStepResult
)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Optional
from abc import ABC, abstractmethod

_registered_multiprocessing_steps = None

def _initialize_worker_process(*, registerd_multiprocessing_steps : dict[int, LinkingStep]):
    global _registered_multiprocessing_steps
    _registered_multiprocessing_steps = registerd_multiprocessing_steps
    for step in _registered_multiprocessing_steps.values():
        step.initialize()

def _worker_process_apply(step_id : int, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
    step = _registered_multiprocessing_steps[step_id]
    return step.apply(address)

def _worker_process_batch_apply(step_id : int, addresses : list[LinkedAddress]) -> list[tuple[LinkedAddress, LinkingStepResult]]:
    step : BatchLinkingStep = _registered_multiprocessing_steps[step_id]
    return step.batch_apply(addresses)


class StepExecutorContext:
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

    def register_multiprocessing_step(self, step : LinkingStep):
        step_id = id(step)
        self.registered_multiprocessing_steps[step_id] = step

    def close(self):
        self.multiprocessing_pool.close()
        self.multiprocessing_pool.join()


class PipelineStepExcutor(ABC):
    def __init__(self, processing_step : LinkingStep, context : StepExecutorContext):
        self.step = processing_step
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






class ProcessingPipeline(ABC):
    def __init__(
            self, 
            max_waiting_jobs : int = 1000, 
            max_running_jobs : int | Literal["auto"] = "auto"
        ):
        self.steps : dict[str, PipelineStep] = {}
    
    def register_step(self, step : 'LinkingStep'):
        self.steps[step.name] = PipelineStep(step)

    

