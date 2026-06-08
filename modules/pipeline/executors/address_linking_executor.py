from abc import ABC, abstractmethod
from typing import Literal, Optional
from modules.pipeline.executors.batch_step_executor import BatchStepExecutor
from modules.pipeline.executors.batch_step_executor import BatchStepExecutor
from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.executors.multiprocessing_step_executor import MultiprocessingStepExecutor
from modules.pipeline.executors.mutex_step_executor import MutexStepExecutor
from modules.pipeline.linking_steps import LinkingStep, ParallelizationType
from modules.pipeline.linking_metadata import AddressLinkingMetadata, LinkingStepMetadata
from modules.pipeline.executors.step_executor import StepExecutor
from modules.pipeline.storage.storage import Storage
import time
import datetime

class _StepWrapper:
    def __init__(self, step : LinkingStep, step_executor : StepExecutor, address_linking_executor : 'AddressLinkingExecutor' = None):
        self.step = step
        self.step_executor = step_executor
        self.address_linking_executor = address_linking_executor

    async def apply(self, address_metadata : AddressLinkingMetadata) -> AddressLinkingMetadata:
        start = datetime.datetime.now().astimezone(datetime.UTC)
        start_monotonic = time.monotonic()
        result_address, step_result = self.step_executor.apply(address_metadata.address)
        elapsed_time_seconds = time.monotonic() - start_monotonic

        step_metadata = LinkingStepMetadata(
            step_name=self.step.name,
            result=step_result,
            start=start,
            elapsed_time_seconds=elapsed_time_seconds
        )
        result = AddressLinkingMetadata(
            address=result_address,
            applied_steps=address_metadata.applied_steps + [step_metadata],
        )

        return result

def default_executor_for_step(step : LinkingStep, context : ExecutorContext) -> StepExecutor:
    if step.parallelization_type == ParallelizationType.NONE:
        return MutexStepExecutor(step, context)
    elif step.parallelization_type == ParallelizationType.SIMPLE:
        return MultiprocessingStepExecutor(step, context)
    elif step.parallelization_type == ParallelizationType.BATCH:
        return BatchStepExecutor(step, context)
    else:
        raise ValueError(f"Unknown parallelization type {step.parallelization_type} for step {step}")

class AddressLinkingExecutor(ABC): 
    def __init__(
            self, 
            *,
            executor_context : Optional[ExecutorContext] = None, 
            num_workers : int | Literal['auto'] = 'auto',
            storage : Storage
        ):
        self.executor_context = executor_context or ExecutorContext(num_workers=num_workers)
        self.storage = storage

    def register_step(self, step : LinkingStep, executor : Optional[StepExecutor]) -> _StepWrapper:
        if executor is None:
            executor = default_executor_for_step(step, self.executor_context)
        return _StepWrapper(step, executor, self)
    
    
    @abstractmethod
    def register_steps(self):
        pass

    def on_step_complete(self, address_metadata : AddressLinkingMetadata, step_metadata : LinkingStepMetadata):
        pass


    @abstractmethod
    async def apply(self, address : AddressLinkingMetadata) -> AddressLinkingMetadata:
        pass

