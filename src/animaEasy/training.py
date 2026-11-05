import collections
import copy
import json
import logging
import os
import pathlib
import secrets
import threading
import time

import filelock
import psutil

from animaEasy.domain import MODEL_ROLES, Project
from animaEasy.errors import ValidationError
from animaEasy.storage import readJson
from animaEasy.trainingArguments import trainingArguments
from animaEasy.trainingAssets import runtimeContract
from animaEasy.trainingProcess import ProcessRunner
from animaEasy.trainingRuns import ACTIVE_STATUSES, RunStore, captureInputs, processAlive, verifyInputs
from animaEasy.trainingRuntime import RuntimeManager


class TrainingController:
    def __init__(self, workspace):
        self.workspace = workspace
        self.runtime = RuntimeManager(workspace.statePath, pathlib.Path(workspace.preferences.runtimeStoragePath) if workspace.preferences.runtimeStoragePath else None)
        self.runner = ProcessRunner()
        self.lock = threading.Lock()
        self.recordLock = threading.Lock()
        self.cancelEvent = threading.Event()
        self.thread = None
        self.closed = False
        self.activePath = workspace.statePath / "training" / "active.json"
        self.stateKeyPath = self.activePath.with_name("state.key")
        self.state = {"busy": False, "kind": "idle", "status": "idle", "message": "No training job is running.", "run": None, "progress": None, "version": 0}
        self.lines = collections.deque(maxlen=220)
        self.store = None
        self.record = None
        self.logPath = None
        self.nextSave = 0

    @property
    def busy(self) -> bool:
        with self.lock:
            return self.state["busy"]

    def snapshot(self) -> dict:
        with self.lock:
            return {**copy.deepcopy(self.state), "lines": tuple(self.lines)}

    def update(self, **values) -> None:
        with self.lock:
            self.state.update(values)
            self.state["version"] += 1

    def appendLine(self, line: str) -> None:
        with self.lock:
            self.lines.append(line[-16384:])
            self.state["version"] += 1

    def message(self, message: str) -> None:
        self.update(message=message)
        self.appendLine(message)
        if self.logPath:
            with self.logPath.open("a", encoding="utf-8") as stream:
                stream.write(f"\n[Anima Easy] {message}\n")

    def saveRecord(self, **values) -> None:
        with self.recordLock:
            self.record.update(values)
            self.store.save(self.record)
            self.update(run=dict(self.record))

    def launch(self, kind: str, operation) -> None:
        with self.lock:
            if self.closed:
                raise ValidationError("The application is closing.")
            if self.state["busy"]:
                raise ValidationError("One managed job can run at a time. Stop the current operation first.")
            self.state.update({"busy": True, "kind": kind, "status": "preparing", "message": "Preparing…", "run": None, "progress": None})
            self.state["version"] += 1
            self.lines.clear()
        self.cancelEvent = threading.Event()
        self.record, self.store, self.logPath = None, None, None

        def work() -> None:
            try:
                self.activePath.parent.mkdir(parents=True, exist_ok=True)
                with filelock.FileLock(self.activePath.with_name("job.lock"), timeout=0):
                    self.runtime.root.parent.mkdir(parents=True, exist_ok=True)
                    with filelock.FileLock(self.runtime.lockPath, timeout=0):
                        if self.activePath.exists():
                            active = runtimeContract.readData(self.activePath)
                            if processAlive(active.get("processId"), active.get("processCreatedAt")):
                                raise ValidationError("The previous managed process is still shutting down. Wait a moment before starting another job.")
                            self.activePath.unlink()
                        self.ensureStateKey()
                        operation()
            except Exception as error:
                canceled = self.cancelEvent.is_set() or isinstance(error, InterruptedError)
                status = "canceled" if canceled else "failed"
                message = "Canceled. Only completed epoch checkpoints can be resumed; unsaved work is lost." if canceled else str(error)
                if isinstance(error, filelock.Timeout):
                    message = "Another application instance owns the managed training runtime. Stop its operation first."
                logging.exception("Managed %s operation ended: %s", kind, message)
                try:
                    self.message(message)
                    if self.record:
                        self.saveRecord(status=status, message=message)
                except Exception:
                    logging.exception("The training result could not be saved")
                self.update(status=status, message=message)
            finally:
                self.update(busy=False)

        self.thread = threading.Thread(target=work, name="Managed training", daemon=True)
        self.thread.start()

    def ensureStateKey(self) -> None:
        identityPath = self.activePath.with_name("stateIdentity.json")
        if self.stateKeyPath.is_symlink() or identityPath.is_symlink():
            raise ValidationError("Trainer-state ownership files cannot be symbolic links.")
        if not self.stateKeyPath.exists():
            if identityPath.exists():
                raise ValidationError("The trainer-state ownership key is missing. Restore the application-state backup before resuming or starting training.")
            descriptor = os.open(self.stateKeyPath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        if self.stateKeyPath.stat().st_size != 32:
            raise ValidationError("The trainer-state ownership key is damaged. It has not been replaced.")
        digest = runtimeContract.fileDigest(self.stateKeyPath)
        if identityPath.exists():
            if runtimeContract.readData(identityPath).get("sha256") != digest:
                raise ValidationError("The trainer-state ownership key changed. Restore the matching application-state backup.")
        else:
            runtimeContract.writeData(identityPath, {"schemaVersion": 1, "sha256": digest})

    def setRuntimeStorage(self, path: str) -> None:
        with self.lock:
            if self.closed or self.state["busy"]:
                raise ValidationError("Stop the managed operation before changing the runtime location.")
            if not isinstance(path, str) or not path.strip():
                raise ValidationError("Choose a runtime storage folder.")
            runtime = RuntimeManager(self.workspace.statePath, pathlib.Path(path))
            runtime.validateStorage()
            self.workspace.setRuntimeStorage(str(runtime.root.parent))
            self.runtime = runtime
            self.state.update({"kind": "idle", "status": "idle", "message": "Runtime location saved. Existing installations have not been moved or deleted.", "run": None, "progress": None})
            self.state["version"] += 1
            self.lines.clear()
            self.record, self.store, self.logPath = None, None, None

    def setup(self) -> None:
        def install() -> None:
            self.runtime.claimStorage()
            self.logPath = self.runtime.logPath
            self.logPath.parent.mkdir(parents=True, exist_ok=True)
            self.message("Setting up the isolated training runtime…")
            result = self.runtime.install(self.runner, self.cancelEvent, self.appendLine)
            self.update(status="completed", message=result["message"])
        self.launch("setup", install)

    def checkHardware(self) -> None:
        if not self.runtime.installed():
            raise ValidationError("Set up the managed runtime before checking CUDA.")
        def check() -> None:
            self.logPath = self.runtime.logPath
            self.message("Checking the managed runtime and CUDA…")
            result = self.runtime.probe(self.runner, self.cancelEvent, self.appendLine)
            self.update(status="completed", message=result["message"])
        self.launch("check", check)

    def start(self, project: Project) -> None:
        if not self.runtime.installed():
            raise ValidationError("Set up the managed training runtime first.")
        project = Project.fromData(project.toData())
        models = {role: self.workspace.models.get(project.modelIds.get(role, "")) for role in MODEL_ROLES}
        projects = self.workspace.projects
        self.launch("training", lambda: self.runTraining(RunStore(projects, self.stateKeyPath), project, models))

    def resume(self, projectId: str, runId: str, stateName: str) -> None:
        if not self.runtime.installed():
            raise ValidationError("Set up the managed training runtime first.")
        store = RunStore(self.workspace.projects, self.stateKeyPath)
        source = store.load(projectId, runId)
        if source["status"] in ACTIVE_STATUSES:
            raise ValidationError("The selected run has not stopped yet.")
        if stateName not in {state["name"] for state in store.states(source)}:
            raise ValidationError("Choose a completed state saved by this application. Imported trainer states are not accepted.")
        recipe = readJson(store.folder(projectId, runId) / "inputs" / "recipe.json")
        project = Project.fromData(recipe["project"])
        self.launch("training", lambda: self.runTraining(store, project, {}, {"runId": runId, "stateName": stateName}))

    def runTraining(self, store: RunStore, project: Project, models: dict, resumeFrom: dict | None = None) -> None:
        self.store = store
        self.record = store.create(project, self.runtime.fingerprint, resumeFrom)
        folder = store.folder(project.id, self.record["id"])
        self.logPath = folder / "training.log"
        self.saveRecord(totalEpochs=project.training.epochs)
        self.message("Checking CUDA before loading training models…")
        hardware = self.runtime.probe(self.runner, self.cancelEvent, self.appendLine, logPath=self.logPath)
        runtimeContract.writeData(folder / "inputs" / "hardware.json", hardware)
        if not hardware.get("cudaReady"):
            raise ValidationError(hardware["message"])
        if project.training.mixedPrecision == "bf16" and not hardware.get("bf16Supported"):
            raise ValidationError("This GPU does not support BF16. Choose FP16 in the training settings and start a new run.")
        resume = None
        if resumeFrom:
            source = store.load(project.id, resumeFrom["runId"])
            sourceFolder = store.folder(project.id, source["id"])
            self.message("Verifying the original run inputs. Current project edits are not used for resume.")
            recipe = verifyInputs(sourceFolder, source["recipeHash"], self.runtime.fingerprint, self.cancelEvent, self.appendLine)
            statePath = sourceFolder / "outputs" / resumeFrom["stateName"]
            verified = runtimeContract.verifyState(statePath, source["id"], source["recipeHash"], self.stateKeyPath.read_bytes(), self.cancelEvent)
            resume = {"runId": source["id"], "path": str(statePath), "completedEpochs": verified["completedEpochs"], "completedSteps": verified["completedSteps"]}
            store.copySnapshot(sourceFolder, folder)
        else:
            self.message("Capturing captions and checking existing image and model files. Model weights stay in place.")
            recipe = captureInputs(folder, project, models, self.runtime.fingerprint, self.cancelEvent, self.appendLine)
        if self.cancelEvent.is_set():
            raise InterruptedError("Training canceled before launch.")
        recipeHash = runtimeContract.fileDigest(folder / "inputs" / "recipe.json")
        self.saveRecord(recipeHash=recipeHash, imageCount=len(recipe["images"]), emptyCaptions=recipe["emptyCaptions"], completedEpochs=resume["completedEpochs"] if resume else 0, completedSteps=resume["completedSteps"] if resume else 0)
        arguments = trainingArguments(folder, recipe, resume)
        store.copyBridge(folder)
        owner = psutil.Process()
        token = secrets.token_hex(24)
        specification = {"schemaVersion": 1, "runId": self.record["id"], "runFolder": str(folder), "recipeHash": recipeHash, "trainerPath": str(self.runtime.trainerPath), "stateKeyPath": str(self.stateKeyPath), "runtimeFingerprint": self.runtime.fingerprint, "arguments": arguments, "resume": resume, "ownerPid": owner.pid, "ownerCreatedAt": owner.create_time(), "token": token}
        specificationPath = folder / "inputs" / "execution.json"
        runtimeContract.writeData(specificationPath, specification)
        command = self.runtime.command(folder / "inputs" / "runtimeBridge.py", str(specificationPath))
        runtimeContract.writeData(folder / "inputs" / "launch.json", {"schemaVersion": 1, "command": command, "workingDirectory": str(self.runtime.trainerPath)})
        completed = threading.Event()

        def started(processId: int, processCreatedAt: float) -> None:
            runtimeContract.writeData(self.activePath, {"schemaVersion": 1, "runFolder": str(folder), "processId": processId, "processCreatedAt": processCreatedAt})
            self.saveRecord(status="running", processId=processId, processCreatedAt=processCreatedAt)
            self.update(status="running")

        def received(line: str) -> None:
            if not line.startswith(runtimeContract.EVENT_PREFIX):
                self.appendLine(line)
                return
            try:
                event = json.loads(line[len(runtimeContract.EVENT_PREFIX):])
                if event.get("token") != token:
                    self.appendLine(line)
                    return
                kind = event.get("kind")
                if kind == "progress":
                    step, total = event.get("step"), event.get("total")
                    if type(step) is not int or type(total) is not int or not 0 <= step <= total or total < 1:
                        raise ValueError("Invalid trainer progress counters")
                    if event["epochs"] != project.training.epochs:
                        raise ValueError("Trainer epoch target changed")
                    self.update(progress=event, message="Training" if not self.cancelEvent.is_set() else "Stopping after the current operation…")
                    if time.monotonic() >= self.nextSave:
                        self.saveRecord(completedSteps=step, totalSteps=total, message="Training", lastLoss=event.get("loss"), currentEpoch=event["epoch"])
                        self.nextSave = time.monotonic() + 2
                elif kind == "phase":
                    self.update(message=event["message"])
                    self.saveRecord(message=event["message"])
                    self.appendLine(event["message"])
                elif kind == "stateSaved":
                    self.saveRecord(completedEpochs=event["completedEpochs"])
                    self.appendLine(f"Trainer state saved: epoch {event['completedEpochs']} — {event['name']}")
                elif kind == "checkpointSaved":
                    self.appendLine(f"LoRA checkpoint saved: {event['name']}")
                elif kind == "completed":
                    if event.get("completedEpochs") != project.training.epochs:
                        raise ValueError("Trainer finished before its epoch target")
                    self.saveRecord(completedSteps=event["completedSteps"], completedEpochs=event["completedEpochs"])
                    completed.set()
                elif kind == "stopped":
                    self.appendLine(event["message"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValidationError(f"The trainer sent an invalid status event: {error}") from error

        self.message("Starting the pinned Anima LoRA trainer. Keep this application open; closing it stops the managed process.")
        try:
            code = self.runner.run(command, self.runtime.trainerPath, self.runtime.environment(), self.logPath, self.cancelEvent, received, started, folder / "stop.request")
        finally:
            self.activePath.unlink(missing_ok=True)
        self.saveRecord(exitCode=code)
        if code == 0 and completed.is_set() and "anima.safetensors" in store.checkpoints(self.record):
            message = "Training completed. Final LoRA weights, trainer state, captured inputs, and the full log are saved."
            self.saveRecord(status="completed", message=message, exitCode=code)
            self.update(status="completed", message=message)
        elif self.cancelEvent.is_set():
            raise InterruptedError("Training stopped.")
        else:
            with self.lock:
                outOfMemory = any("out of memory" in line.lower() for line in self.lines)
            suggestion = " GPU memory was exhausted. For a new run, try a lower resolution, batch size 1, caching, gradient checkpointing, and more swapped blocks." if outOfMemory else " Read the full run log for the trainer error."
            self.saveRecord(exitCode=code)
            raise ValidationError(f"Training ended without a verified final result (exit {code}).{suggestion}")

    def cancel(self) -> None:
        if self.busy:
            self.cancelEvent.set()
            self.update(status="stopping", message="Stopping… Completed epoch states are kept; unsaved work will be lost.")

    def close(self) -> None:
        self.closed = True
        self.cancel()
        if self.thread is not None:
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                self.runner.terminate()
                self.thread.join(timeout=5)
