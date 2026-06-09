from abc import ABC, abstractmethod
from typing import Literal, Optional

import asyncio
from modules.pipeline.executors.batch_step_executor import BatchStepExecutor
from modules.pipeline.executors.batch_step_executor import BatchStepExecutor
from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.executors.multiprocessing_step_executor import MultiprocessingStepExecutor
from modules.pipeline.executors.mutex_step_executor import MutexStepExecutor
from modules.pipeline.linking_steps import LinkingStep, ParallelizationType, LinkingStepResult, Failed
from modules.pipeline.linking_metadata import AddressLinkingMetadata, LinkingStepMetadata
from modules.pipeline.executors.step_executor import StepExecutor
from modules.pipeline.storage.storage import Storage
import time
import datetime
import logging
import traceback
from io import StringIO
from tqdm.auto import tqdm

class _SkipProcessingSignal(Exception):
    pass

class _StepWrapper:
    def __init__(self, step : LinkingStep, step_executor : StepExecutor, address_linking_executor : 'AddressLinkingExecutor' = None):
        self.step = step
        self.step_executor = step_executor
        self.address_linking_executor = address_linking_executor

    async def apply(self, address_metadata : AddressLinkingMetadata) -> AddressLinkingMetadata:
        start = datetime.datetime.now().astimezone(datetime.UTC)
        start_monotonic = time.monotonic()
        exception_throw = None
        try:
            result_address, step_result = await self.step_executor.apply(address_metadata.address)
        except Exception as e:
            self.address_linking_executor.logger.exception(f"Error processing address {address_metadata.address.id}: {e}")
            exception_throw = e
            result_address = address_metadata.address
            trace_buffer = StringIO()
            traceback.print_exception(type(e), e, e.__traceback__, file=trace_buffer)
            step_result = Failed(
                error_message=str(e),
                exception_class=e.__class__.__name__,
                traceback=trace_buffer.getvalue()
            )

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

        if exception_throw is None:
            await self.address_linking_executor.on_step_complete(result, step_metadata)
        else:
            await self.address_linking_executor.on_step_failed(result, step_metadata)
        return result

def default_executor_for_step(step : LinkingStep) -> StepExecutor:
    if step.parallelization_type == ParallelizationType.NONE:
        return MutexStepExecutor(step)
    elif step.parallelization_type == ParallelizationType.SIMPLE:
        return MultiprocessingStepExecutor(step)
    elif step.parallelization_type == ParallelizationType.BATCH:
        return BatchStepExecutor(step)
    else:
        raise ValueError(f"Unknown parallelization type {step.parallelization_type} for step {step}")

class AddressLinkingExecutor(ABC): 
    def __init__(
            self, 
            *,
            executor_context : Optional[ExecutorContext] = None,
            num_workers : int | Literal['auto'] = 'auto',
            max_concurrent_tasks : int = 100,
            monitor_log_interval : float | datetime.timedelta = 60.0,
            logger : Optional[logging.Logger] = None,
            storage : Storage
        ):
        self.executor_context = executor_context or ExecutorContext(
            num_processes=num_workers, max_concurrent_tasks=max_concurrent_tasks)
        self.storage = storage
        if isinstance(monitor_log_interval, datetime.timedelta):
            monitor_log_interval = monitor_log_interval.total_seconds()
        self.monitor_log_interval = monitor_log_interval
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def register_step(self, step : LinkingStep, executor : Optional[StepExecutor] = None) -> _StepWrapper:
        if executor is None:
            executor = default_executor_for_step(step, self.executor_context)
        else:
            assert step.name == executor.step.name, "Step name must match between the step and the executor"
        return _StepWrapper(step, executor, self)
    
    
    @abstractmethod
    def register_steps(self):
        pass

    async def on_step_complete(
            self, 
            address_metadata : AddressLinkingMetadata, 
            step_metadata : LinkingStepMetadata
        ):
        await self.storage.upsert(address_metadata)

    async def on_step_failed(
            self,
            address_metadata : AddressLinkingMetadata, 
            step_metadata : LinkingStepMetadata,
        ):
        await self.storage.upsert(address_metadata, preserve_linking_data=True)


    async def monitor(self):
        while True:
            await asyncio.sleep(self.monitor_log_interval)
            # TODO: Implement monitoring logic

    async def _dispatcher(self):
        while True:
            address_metadata = await self.storage.fetch_pending()
            if address_metadata is None:
                # TODO set up drawing more addresses? 
                # Current assumption is that storage contains all pending addresses already
                break
            try:
                result = await self.apply(address_metadata)
            except _SkipProcessingSignal:
                continue
            except Exception as e:
                self.logger.exception(f"Uncaught error processing address {address_metadata.address.id}: {e}")
                
            result.finished = True
            await self.storage.upsert(result)

    
    async def run_async(self):
        self.executor_context.initialize()
        self.register_steps()
        for step_wrapper in self.executor_context.get_all_step_wrappers():
            step_wrapper.step_executor.initialize()
        
        with asyncio.TaskGroup() as tg:
            for _ in range(self.executor_context.num_processes):
                tg.create_task(self._dispatcher())
        

    @abstractmethod
    async def apply(self, address : AddressLinkingMetadata) -> AddressLinkingMetadata:
        pass

