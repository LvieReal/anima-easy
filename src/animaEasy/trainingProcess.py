import codecs
import ctypes
import ctypes.wintypes
import os
import pathlib
import re
import signal
import subprocess
import threading
import time

import psutil


ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class WindowsJob:
    def __init__(self, processId: int):
        class BasicLimits(ctypes.Structure):
            _fields_ = [("processTime", ctypes.c_int64), ("jobTime", ctypes.c_int64), ("flags", ctypes.wintypes.DWORD), ("minimumWorkingSet", ctypes.c_size_t), ("maximumWorkingSet", ctypes.c_size_t), ("activeProcesses", ctypes.wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority", ctypes.wintypes.DWORD), ("scheduling", ctypes.wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("readOperations", "writeOperations", "otherOperations", "readBytes", "writeBytes", "otherBytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("processMemory", ctypes.c_size_t), ("jobMemory", ctypes.c_size_t), ("peakProcessMemory", ctypes.c_size_t), ("peakJobMemory", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (("CreateJobObjectW", [ctypes.c_void_p, ctypes.c_wchar_p], ctypes.wintypes.HANDLE), ("SetInformationJobObject", [ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.wintypes.DWORD], ctypes.wintypes.BOOL), ("OpenProcess", [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD], ctypes.wintypes.HANDLE), ("AssignProcessToJobObject", [ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE], ctypes.wintypes.BOOL), ("CloseHandle", [ctypes.wintypes.HANDLE], ctypes.wintypes.BOOL)):
            function = getattr(self.kernel, name)
            function.argtypes = arguments
            function.restype = result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            processHandle = self.kernel.OpenProcess(0x0101, False, processId)
            if not processHandle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not self.kernel.AssignProcessToJobObject(self.handle, processHandle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self.kernel.CloseHandle(processHandle)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class ProcessRunner:
    def __init__(self):
        self.process = None
        self.job = None
        self.lock = threading.Lock()

    def terminate(self) -> None:
        with self.lock:
            process = self.process
            if process is None:
                return
            if self.job is not None:
                self.job.close()
                self.job = None
            elif os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()

    def run(self, command: list[str], cwd: pathlib.Path, environment: dict[str, str], logPath: pathlib.Path, cancel: threading.Event, onLine=None, onStarted=None, stopPath: pathlib.Path | None = None, graceSeconds: float = 20, timeoutSeconds: float | None = None) -> int:
        if cancel.is_set():
            raise InterruptedError("Canceled before launch.")
        logPath.parent.mkdir(parents=True, exist_ok=True)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004} if os.name == "nt" else {"start_new_session": True}
        errors = []
        with logPath.open("ab", buffering=0) as log:
            with self.lock:
                if self.process is not None:
                    raise RuntimeError("A managed process is already running.")
                process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, **options)
                self.process = process
                try:
                    if os.name == "nt":
                        self.job = WindowsJob(process.pid)
                        psutil.Process(process.pid).resume()
                except BaseException:
                    if self.job is not None:
                        self.job.close()
                        self.job = None
                    process.kill()
                    process.wait()
                    self.process = None
                    raise
            def readOutput():
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                pending = ""
                try:
                    while chunk := process.stdout.read(16384):
                        if log.write(chunk) != len(chunk):
                            raise OSError("The full process log could not be written. Check available disk space.")
                        pending += decoder.decode(chunk)
                        pieces = re.split(r"[\r\n]", pending)
                        pending = pieces.pop()
                        for line in pieces:
                            if line and onLine:
                                onLine(ANSI_PATTERN.sub("", line)[-16384:])
                        while len(pending) > 16384:
                            if onLine:
                                onLine(ANSI_PATTERN.sub("", pending[:16384]))
                            pending = pending[16384:]
                    pending += decoder.decode(b"", final=True)
                    if pending and onLine:
                        onLine(ANSI_PATTERN.sub("", pending))
                except BaseException as error:
                    errors.append(error)

            reader = threading.Thread(target=readOutput, name="Training output", daemon=True)
            reader.start()
            startedAt = time.monotonic()
            cancellationAt = None
            expired = False
            try:
                if onStarted:
                    try:
                        createdAt = psutil.Process(process.pid).create_time()
                    except psutil.NoSuchProcess:
                        createdAt = None
                    onStarted(process.pid, createdAt)
                while process.poll() is None:
                    expired = timeoutSeconds is not None and time.monotonic() - startedAt > timeoutSeconds
                    if errors or expired:
                        self.terminate()
                    if cancel.is_set():
                        if cancellationAt is None:
                            cancellationAt = time.monotonic()
                            if stopPath is not None:
                                stopPath.touch(exist_ok=True)
                            else:
                                self.terminate()
                        if time.monotonic() - cancellationAt >= graceSeconds:
                            self.terminate()
                    time.sleep(0.05)
                if expired:
                    raise TimeoutError("The managed process exceeded its time limit.")
                returnCode = process.wait()
            finally:
                self.terminate()
                process.wait()
                reader.join(timeout=5)
                process.stdout.close()
                with self.lock:
                    self.process = None
                if reader.is_alive():
                    errors.append(RuntimeError("The process output stream did not close."))
            if errors:
                raise RuntimeError(f"Cannot capture training output: {errors[0]}") from errors[0]
            return returnCode


def trainingEnvironment() -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if name not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} and not name.startswith(("ACCELERATE_", "OMPI_", "PMI_", "PMIX_", "MV2_"))}
    environment.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "2"})
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return environment
