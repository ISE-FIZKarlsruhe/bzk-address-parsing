
from typing import Optional

from modules.pipeline.executors.executor_context import ExecutorContext
from modules.pipeline.linked_data import LinkedAddress
from dataclasses import dataclass
import threading
import queue
from modules.pipeline.executors.step_executor import StepExecutor
from modules.pipeline.linking_steps import LinkingStepResult, BatchLinkingStep
import asyncio
import time

@dataclass
class Job:
    input_data : LinkedAddress
    result : Optional[tuple[LinkedAddress, LinkingStepResult]] = None
    finished : asyncio.Event

# TODO correct this whole logic to use asyncio properly


class BatchGatheringThread:
    def __init__(self, step : BatchLinkingStep, loop : asyncio.AbstractEventLoop):
        self.step = step
        self.batch_size = step.batch_size
        self.batch_gathering_patience = step.batch_gathering_patience or 10.0
        self.current_batch_size = 0
        self.queue : queue.Queue[Job] = queue.Queue(step.batch_size)        
        self.thread = threading.Thread(
            name=f"BatchGatheringThread-{step.name}",
            target=self._run_thread,
            daemon=True
        )
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
        batch_gathering_deadline = time.monotonic() + self.batch_gathering_patience
        while len(batch) < self.batch_size:
            try:
                job = self.queue.get(timeout=max(batch_gathering_deadline-time.monotonic(), 0.1))
                if self.queue_full.is_set():
                    self.asyncio_event_loop.call_soon_threadsafe(self.queue_full.clear)
                if job is None:
                    if len(batch) == 0:
                        raise queue.ShutDown()
                    else:
                        break
                self.current_batch_size += 1
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
                self.current_batch_size = 0
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
        self.rate = 0.0
    
    def initialize(self):
        self.gathering_thread.start()

    def get_pending_count(self):
        return self.gathering_thread.queue.qsize() + self.gathering_thread.current_batch_size
    
    def get_rate(self) -> float:
        return self.rate

    async def apply(self, address : LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
        job = Job(input_data=address, finished=asyncio.Event())
        start = time.monotonic()
        await self.gathering_thread.submit_job(job)
        await job.finished.wait()
        elapsed = time.monotonic() - start
        self.rate = 1 / elapsed if elapsed > 0 else float("inf")
        assert job.result is not None
        return job.result
    
    def finalize(self):
        self.gathering_thread.stop()
