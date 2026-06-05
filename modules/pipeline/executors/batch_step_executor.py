
from typing import Optional

from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.linked_data import LinkedAddress
from dataclasses import dataclass
import threading
import queue
from modules.pipeline.executors.step_executor import StepExecutor
from modules.pipeline.linking_steps import LinkingStepResult, BatchLinkingStep
import asyncio

@dataclass
class Job:
    input_data : LinkedAddress
    result : Optional[tuple[LinkedAddress, LinkingStepResult]] = None
    finished : asyncio.Event


class BatchGatheringThread:
    def __init__(self, step : BatchLinkingStep, loop : asyncio.AbstractEventLoop):
        self.step = step
        self.batch_size = step.batch_size
        self.gather_timeout_seconds = step.estimated_processing_time or 10.0
        self.queue : queue.Queue[Job] = queue.Queue(step.batch_size)        
        self.thread = threading.Thread(target=self._run_thread, args=(step, self.queue), daemon=True)
        self.asyncio_event_loop = loop

        # Set by queueing threads when the queue is full
        # Used to enable awaiting queue space with futures API
        self.queue_full = asyncio.Event()
        # Set by the gathering thread when it is processing a batch
        # Used to enable try_submit_job to avoid assigning tasks to a busy resource
        self.busy = threading.Event()
        # Set to stop gathering thread
        self.killed = threading.Event()

    def _gather_batch(self) -> tuple[list[Job], list[LinkedAddress]]:
        jobs = []
        batch = []
        while len(batch) < self.batch_size:
            try:
                job = self.queue.get(timeout=self.gather_timeout_seconds)
                if self.queue_full.is_set():
                    self.asyncio_event_loop.call_soon_threadsafe(self.queue_full.clear)
                if job is None:
                    if len(batch) == 0:
                        raise queue.ShutDown()
                    else:
                        break
                jobs.append(job)
                batch.append(job.input_data)
            except queue.Empty:
                if len(batch) > 0:
                    break
            except queue.ShutDown:
                if len(batch) == 0:
                    raise
                else:
                    break
        return jobs, batch

    def _run_thread(self):
        self.step.initialize()
        try:
            while True:
                jobs, batch = self._gather_batch()
                if len(batch) == 0:
                    continue
                self.busy.set()
                results = self.step.batch_apply(batch)
                for job, result in zip(jobs, results):
                    job.result = result
                    self.asyncio_event_loop.call_soon(job.finished.set)
                    self.queue.task_done()
                self.busy.clear()
        except queue.ShutDown:
            pass
        self.step.finalize()

    async def submit_job(self, job : Job):
        job_queued = False
        while not job_queued:
            try:
                self.queue.put(job, block=False)
                job_queued = True
            except queue.Full:
                self.queue_full.set()
                if self.queue.full():
                    await self.queue_full.wait()
                else:
                    self.queue_full.clear()

    def try_submit_job(self, job : Job) -> bool:
        """
        Tries to submit a job without blocking, returns True is successful.
        
        May fail even if the queue is not technically full, 
        if the gathering thread is busy processing a batch and the queue is only accumulating pending jobs.
        """
        if self.busy.is_set():
            return False
        try:
            self.queue.put(job, block=False)
            return True 
        except queue.Full:
            return False

    def start(self):
        self.thread.start()

    def stop(self):
        self.killed.set()
        self.thread.join()

class BatchStepExecutor(StepExecutor):
    def __init__(self, processing_step : BatchLinkingStep, context : ExecutorContext):
        super().__init__(processing_step, context)
        self.gathering_thread = BatchGatheringThread(processing_step, asyncio.get_event_loop())
    
    def initialize(self):
        self.gathering_thread.start()

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        job = Job(input_data=address, finished=asyncio.Event())
        await self.gathering_thread.submit_job(job)
        await job.finished.wait()
        assert job.result is not None
        return job.result
    
    def finalize(self):
        self.gathering_thread.stop()


class RoundRobinBatchStepExecutor(StepExecutor):
    def __init__(self, child_executors : list[BatchStepExecutor], context : ExecutorContext):
        super().__init__(context)
        self.child_executors = child_executors
        self.current_index = 0

    def initialize(self):
        for executor in self.child_executors:
            executor.initialize()

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        # Round-robin distribution when child executors are busy
        start_index = self.current_index
        job = Job(input_data=address, finished=asyncio.Event())
        while True:
            executor = self.child_executors[self.current_index]
            if executor.gathering_thread.try_submit_job(job):
                break
            else:
                self.current_index = (self.current_index + 1) % len(self.child_executors)
                if self.current_index == start_index:
                    # All executors are busy, wait for the current one to finish
                    await executor.gathering_thread.submit_job(job)
                    break
        await job.finished.wait()
        assert job.result is not None
        return job.result

    def finalize(self):
        for executor in self.child_executors:
            executor.finalize()