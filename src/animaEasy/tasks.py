import concurrent.futures
import dataclasses
import threading
from collections.abc import Callable


@dataclasses.dataclass
class PendingTask:
    future: concurrent.futures.Future
    onSuccess: Callable
    onError: Callable


class TaskQueue:
    def __init__(self):
        self.cancelEvent = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="AnimaEasy")
        self.pending: list[PendingTask] = []

    @property
    def busy(self) -> bool:
        return bool(self.pending)

    def submit(self, action: Callable, onSuccess: Callable, onError: Callable) -> concurrent.futures.Future:
        future = self.executor.submit(action)
        self.pending.append(PendingTask(future, onSuccess, onError))
        return future

    def poll(self) -> None:
        finished = []
        remaining = []
        for task in self.pending:
            (finished if task.future.done() else remaining).append(task)
        self.pending = remaining
        for task in finished:
            try:
                result = task.future.result()
            except Exception as error:
                task.onError(error)
            else:
                task.onSuccess(result)

    def close(self) -> None:
        self.cancelEvent.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.poll()
