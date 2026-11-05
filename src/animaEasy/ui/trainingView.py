import datetime
import json
import os
import pathlib
import subprocess
import time

from dearpygui import dearpygui

from animaEasy.errors import ValidationError
from animaEasy.trainingAssets import runtimeContract
from animaEasy.trainingProcess import ANSI_PATTERN
from animaEasy.trainingRuns import ACTIVE_STATUSES, RunStore
from animaEasy.trainingRuntime import TRAINER_REVISION
from animaEasy.ui import theme, widgets


def openFolder(folder: pathlib.Path) -> None:
    if not folder.is_dir():
        raise ValidationError("This folder has not been created yet.")
    if os.name == "nt":
        os.startfile(folder)
    else:
        subprocess.Popen(["xdg-open", str(folder)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def logTail(path: pathlib.Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 262144))
            lines = []
            for line in stream.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines():
                line = ANSI_PATTERN.sub("", line)
                if line.startswith(runtimeContract.EVENT_PREFIX):
                    try:
                        event = json.loads(line[len(runtimeContract.EVENT_PREFIX):])
                        if isinstance(event, dict) and isinstance(event.get("message", ""), str):
                            line = event.get("message", "")
                    except (ValueError, TypeError):
                        pass
                if line:
                    lines.append(line)
            return "\n".join(lines[-220:])
    except FileNotFoundError:
        return "No log has been written yet."


class TrainingView:
    section = "training"

    def __init__(self, shell, projectId: str | None = None):
        self.shell = shell
        self.projectId = projectId
        self.store = RunStore(shell.workspace.projects, shell.training.stateKeyPath)
        self.selected = None
        self.records = {}
        self.stateChoices = {}
        self.nextUpdate = 0
        self.runtimeStoragePath = str(shell.training.runtime.root.parent)
        self.lastVersion = -1
        self.wasBusy = False
        self.followOperation = True

    def draw(self) -> None:
        controller = self.shell.training
        widgets.pageHeading("TRAINING", "Train locally. Keep your results.", "One managed job at a time. Every run keeps its captured settings, captions, checkpoints, and full log.")
        with dearpygui.collapsing_header(label="Managed runtime", tag="runtimeSetupPanel", default_open=not controller.runtime.probeResult().get("cudaReady", False)):
            dearpygui.add_text("", tag="managedRuntimeStatus", wrap=0)
            dearpygui.add_text(f"Pinned sd-scripts {TRAINER_REVISION[:8]} • Python 3.12 • PyTorch 2.9.1 / CUDA 12.8", color=theme.colors.muted, wrap=0)
            widgets.pathField(self.shell, "Runtime storage folder", "trainingRuntimePath", self.runtimeStoragePath, True, self.captureRuntimeStorage)
            with dearpygui.tooltip("trainingRuntimePath"):
                dearpygui.add_text("Choose a parent storage folder, such as D:\\Anima training or E:\\Anima training on Windows. The app manages anima-v1 inside it, including Python, downloads, and temporary files. Use folder saves the location; Set up runtime also applies this field before confirmation. Existing installations are not moved or deleted.", wrap=450)
            dearpygui.add_text("", tag="runtimeInstallPath", color=theme.colors.muted, wrap=0)
            dearpygui.add_text("Choose a storage drive before setup. Changing this folder does not move or delete an existing installation.", color=theme.colors.muted, wrap=0)
            dearpygui.add_text("Setup downloads several GB into a separate environment. Keep 12 GiB free, or 24 GiB on drives requiring copies, plus room for projects and checkpoints. Model weights and NVIDIA drivers are not installed or replaced.", color=theme.colors.muted, wrap=0)
            with dearpygui.group(horizontal=True):
                dearpygui.add_button(label="Use folder", tag="applyRuntimeStorage", height=36, callback=self.applyRuntimeStorage)
                widgets.primaryButton("Set up runtime", tag="setupTrainingRuntime", callback=self.setup)
                dearpygui.add_button(label="Check GPU", tag="checkTrainingGpu", height=36, callback=controller.checkHardware)
                dearpygui.add_button(label="Open setup folder", height=36, callback=lambda: openFolder(controller.runtime.root))
        dearpygui.add_spacer(height=6)
        with dearpygui.group(horizontal=True):
            dearpygui.add_button(label="Stop operation", tag="stopTrainingJob", height=32, enabled=False, callback=self.stop)
            dearpygui.add_text("No job is running.", tag="trainingOperation", wrap=0)
        dearpygui.add_progress_bar(tag="trainingProgress", default_value=0, width=-1, overlay="No step progress yet")
        dearpygui.add_text("", tag="trainingProgressDetail", color=theme.colors.muted, wrap=0)
        dearpygui.add_text("Keep Anima Easy open while training. Stop or close discards work since the last completed epoch state.", color=theme.colors.muted, wrap=0)
        dearpygui.add_spacer(height=6)
        hasHistory = bool(self.store.history(self.projectId)[0])
        with dearpygui.collapsing_header(label="Saved runs and checkpoints", tag="runHistoryPanel", default_open=hasHistory and not controller.busy):
            with dearpygui.group(horizontal=True):
                dearpygui.add_text("Run history")
                dearpygui.add_button(label="Refresh", callback=self.reloadHistory)
                dearpygui.add_button(label="All projects", show=self.projectId is not None, callback=lambda: self.shell.showTraining())
            dearpygui.add_combo([], tag="runHistoryChoice", width=-1, callback=self.selectRun)
            dearpygui.add_text("", tag="runHistoryErrors", color=theme.colors.warning, wrap=0)
            dearpygui.add_text("Start a run from a project's Review step.", tag="runSummary", color=theme.colors.muted, wrap=0)
            with dearpygui.group(horizontal=True):
                dearpygui.add_button(label="Open project", tag="openRunProject", enabled=False, callback=self.openProject)
                dearpygui.add_button(label="Outputs / full log", tag="openRunFolder", enabled=False, callback=self.openRunFolder)
                dearpygui.add_button(label="Captured inputs", tag="openRunInputs", enabled=False, callback=self.openInputs)
            with dearpygui.group(horizontal=True):
                dearpygui.add_combo([], tag="resumeStateChoice", width=-190)
                dearpygui.add_button(label="Resume saved epoch", tag="resumeTrainingRun", width=180, enabled=False, callback=self.resume)
            dearpygui.add_text("Resume restores saved model, optimizer, scheduler, and random-generator state. It uses the original epoch target and captured inputs—not current project edits.", color=theme.colors.muted, wrap=0)
        dearpygui.add_spacer(height=6)
        with dearpygui.group(horizontal=True):
            dearpygui.add_text("Output log • latest 220 lines")
            dearpygui.add_checkbox(label="Follow output", tag="followTrainingLog", default_value=True)
            dearpygui.add_button(label="Copy visible log", callback=lambda: dearpygui.set_clipboard_text(dearpygui.get_value("trainingLog")))
        with dearpygui.child_window(tag="trainingLogPanel", width=-1, height=205, border=True, horizontal_scrollbar=True):
            dearpygui.add_text("", tag="trainingLog")
        self.reloadHistory()
        self.tick(force=True)

    def captureRuntimeStorage(self, sender, appData) -> None:
        self.runtimeStoragePath = appData

    def applyRuntimeStorage(self) -> None:
        path = dearpygui.get_value("trainingRuntimePath")
        self.shell.training.setRuntimeStorage(path)
        self.runtimeStoragePath = str(self.shell.training.runtime.root.parent)
        dearpygui.set_value("trainingRuntimePath", self.runtimeStoragePath)
        self.tick(force=True)

    def setup(self) -> None:
        self.applyRuntimeStorage()
        message = f"Install into {self.shell.training.runtime.root}? The managed Python and download cache also stay there.\n\nDownload the pinned sd-scripts trainer and locked PyTorch/CUDA packages into an isolated Python environment? This can take time and several GB. Your model weights, datasets, and UI environment stay in place. NVIDIA drivers are not installed."
        self.shell.confirm("Set up managed training", message, self.shell.training.setup, "Set up runtime")

    def stop(self) -> None:
        self.shell.confirm("Stop the current operation?", "The managed process will stop. Completed epoch checkpoints and their trainer states are kept. Work since the last saved epoch is not recoverable. A partial runtime setup can be repaired by running setup again.", self.shell.training.cancel, "Stop operation")

    def reloadHistory(self) -> None:
        records, errors = self.store.history(self.projectId)
        self.records = {}
        for record in records:
            createdAt = datetime.datetime.fromisoformat(record["createdAt"]).astimezone().strftime("%b %d %H:%M")
            label = f"{createdAt} • {record['projectName']} • {record['status']} • {record['id'][:6]}"
            self.records[label] = record
        active = self.shell.training.snapshot().get("run")
        if self.selected is None and active is not None and (self.projectId is None or active["projectId"] == self.projectId):
            self.selected = active["id"]
        if self.selected is None and records:
            self.selected = records[0]["id"]
        selectedLabel = next((label for label, record in self.records.items() if record["id"] == self.selected), "")
        dearpygui.configure_item("runHistoryChoice", items=list(self.records), enabled=bool(records))
        dearpygui.set_value("runHistoryChoice", selectedLabel)
        dearpygui.set_value("runHistoryErrors", f"{len(errors)} history record(s) could not be read. Existing files have not been replaced." if errors else "")
        self.updateSelection()

    def selectedRecord(self) -> dict | None:
        return next((record for record in self.records.values() if record["id"] == self.selected), None)

    def selectRun(self, sender, appData) -> None:
        self.followOperation = False
        self.selected = self.records[appData]["id"]
        self.updateSelection()
        self.tick(force=True)

    def updateSelection(self) -> None:
        record = self.selectedRecord()
        for tag in ("openRunProject", "openRunFolder", "openRunInputs"):
            dearpygui.configure_item(tag, enabled=record is not None)
        self.stateChoices = {}
        if record:
            states = self.store.states(record)
            target = record.get("totalEpochs", 0)
            self.stateChoices = {f"Epoch {state['completedEpochs']} / {target} • {state['completedSteps']:,} steps": state["name"] for state in states if 0 < state["completedEpochs"] < target}
            checkpoints = self.store.checkpoints(record)
            detail = f"{record['status'].capitalize()} • {record.get('imageCount', '—')} images • {record.get('completedSteps', 0):,} steps • {len(checkpoints)} saved LoRA file(s) • {len(states)} complete state(s)\n{record.get('message', '')}"
            dearpygui.set_value("runSummary", detail)
            dearpygui.set_value("trainingLog", logTail(self.store.folder(record["projectId"], record["id"]) / "training.log"))
        previous = dearpygui.get_value("resumeStateChoice")
        dearpygui.configure_item("resumeStateChoice", items=list(self.stateChoices), enabled=bool(self.stateChoices))
        dearpygui.set_value("resumeStateChoice", previous if previous in self.stateChoices else next(iter(self.stateChoices), "No unfinished epoch state available"))
        self.updateResumeButton()

    def updateResumeButton(self) -> None:
        record = self.selectedRecord()
        enabled = record is not None and record["status"] not in ACTIVE_STATUSES and bool(self.stateChoices) and not self.shell.training.busy and self.shell.training.runtime.installed()
        dearpygui.configure_item("resumeTrainingRun", enabled=enabled)

    def openProject(self) -> None:
        record = self.selectedRecord()
        if record:
            self.shell.openProject(record["projectId"])

    def openRunFolder(self) -> None:
        record = self.selectedRecord()
        if record:
            openFolder(self.store.folder(record["projectId"], record["id"]))

    def openInputs(self) -> None:
        record = self.selectedRecord()
        if record:
            openFolder(self.store.folder(record["projectId"], record["id"]) / "inputs")

    def resume(self) -> None:
        record = self.selectedRecord()
        label = dearpygui.get_value("resumeStateChoice")
        if record and label in self.stateChoices:
            stateName = self.stateChoices[label]
            def accepted() -> None:
                self.shell.training.resume(record["projectId"], record["id"], stateName)
                self.selected = None
            self.shell.confirm("Resume captured run?", f"Resume from {label}? The original inputs will be verified by SHA-256. Current project changes are ignored. A new run folder preserves the earlier logs and outputs.", accepted, "Resume saved epoch")

    def tick(self, force: bool = False) -> None:
        if not force and time.monotonic() < self.nextUpdate:
            return
        self.nextUpdate = time.monotonic() + 0.4
        controller = self.shell.training
        snapshot = controller.snapshot()
        installed = controller.runtime.installed()
        hardware = controller.runtime.probeResult()
        status = hardware.get("message", "Runtime installed. Choose Check GPU to check CUDA.") if installed else "Not installed, incomplete, or awaiting a runtime update."
        dearpygui.set_value("managedRuntimeStatus", status)
        dearpygui.configure_item("managedRuntimeStatus", color=theme.colors.accent if hardware.get("cudaReady") else theme.colors.warning)
        for tag in ("trainingRuntimePath", "trainingRuntimePathBrowse", "applyRuntimeStorage"):
            dearpygui.configure_item(tag, enabled=not snapshot["busy"])
        dearpygui.set_value("runtimeInstallPath", f"Runtime: {controller.runtime.root}")
        dearpygui.configure_item("setupTrainingRuntime", enabled=not snapshot["busy"], label="Repair setup" if installed else "Set up runtime")
        dearpygui.configure_item("checkTrainingGpu", enabled=installed and not snapshot["busy"])
        dearpygui.configure_item("stopTrainingJob", enabled=snapshot["busy"] and snapshot["status"] != "stopping")
        activeName = snapshot["run"]["projectName"] + " • " if snapshot.get("run") else ""
        dearpygui.set_value("trainingOperation", activeName + snapshot["message"])
        progress = snapshot["progress"]
        if progress:
            dearpygui.set_value("trainingProgress", progress["step"] / progress["total"])
            dearpygui.configure_item("trainingProgress", overlay=f"{progress['step']:,} / {progress['total']:,} steps")
            loss = f" • loss {progress['loss']:.5f}" if progress.get("loss") is not None else ""
            dearpygui.set_value("trainingProgressDetail", f"Epoch {progress['epoch']} / {progress['epochs']}{loss}")
        else:
            dearpygui.set_value("trainingProgress", 0)
            dearpygui.configure_item("trainingProgress", overlay="No step progress yet" if snapshot["busy"] else "No active training progress")
            dearpygui.set_value("trainingProgressDetail", "Loading models and caches does not count as completed training steps." if snapshot["busy"] else "")
        active = snapshot.get("run")
        visibleActive = active is not None and (self.projectId is None or active["projectId"] == self.projectId)
        if visibleActive and (self.selected is None or not any(record["id"] == active["id"] for record in self.records.values())):
            self.selected = active["id"]
            self.reloadHistory()
        if snapshot["busy"] and not self.wasBusy:
            self.followOperation = True
        if self.wasBusy and not snapshot["busy"]:
            self.reloadHistory()
            if visibleActive:
                dearpygui.set_value("runHistoryPanel", True)
        if force or snapshot["version"] != self.lastVersion:
            if self.followOperation and (snapshot["kind"] in ("setup", "check") or active and active["id"] == self.selected):
                dearpygui.set_value("trainingLog", "\n".join(snapshot["lines"]))
            if active and active["id"] == self.selected:
                saved = self.store.load(active["projectId"], active["id"])
                for label, record in self.records.items():
                    if record["id"] == active["id"]:
                        changed = any(record.get(name) != saved.get(name) for name in ("completedEpochs", "completedSteps", "status", "message"))
                        self.records[label] = saved
                        if changed:
                            self.updateSelection()
                            if self.followOperation:
                                dearpygui.set_value("trainingLog", "\n".join(snapshot["lines"]))
                self.updateResumeButton()
        self.updateResumeButton()
        if dearpygui.get_value("followTrainingLog"):
            dearpygui.set_y_scroll("trainingLogPanel", dearpygui.get_y_scroll_max("trainingLogPanel"))
        self.wasBusy = snapshot["busy"]
        self.lastVersion = snapshot["version"]
