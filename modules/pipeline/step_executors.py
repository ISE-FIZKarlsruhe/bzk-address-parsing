import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal




class StepExecutor(ABC):

    @abstract

    @abstractmethod
    async def execute_step(self, step):
        pass