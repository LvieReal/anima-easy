import dataclasses
import math
import time

from dearpygui import dearpygui

from animaEasy.captions import CAPTION_MODES
from animaEasy.datasets import inspectDataset
from animaEasy.domain import MODEL_ROLES
from animaEasy.errors import ValidationError
from animaEasy.ui import theme, widgets
from animaEasy.ui.captionView import CaptionView
from animaEasy.ui.projectsView import STEPS
from animaEasy.ui.trainingForm import drawTrainingForm, presetDescription


class EditorView:
    section = "projects"

    def __init__(self, shell, project):
        self.shell = shell
        self.project = project
        self.report = None
        self.scanning = False
        self.dirtyAt = None
        self.saveFailed = False
        self.modelOptions = {}
        self.captionRevision = 0
        self.nextTrainingCheck = 0
        self.captionView = CaptionView(self)

    def draw(self) -> None:
        widgets.pageHeading("PROJECT  /  SAVED DRAFT", self.project.name, "Work through the steps at your own pace. Valid changes are saved automatically.")
        with dearpygui.group(horizontal=True):
            for index, label in enumerate(STEPS):
                item = dearpygui.add_button(label=f"{index + 1:02d}  {label}", tag=f"stepButton{index}", width=self.shell.stepWidth(), height=36, callback=self.selectStep, user_data=index)
                if index == self.project.currentStep:
                    dearpygui.bind_item_theme(item, "selectedTheme")
        dearpygui.add_spacer(height=4)
        with dearpygui.child_window(tag="projectBody", height=-70, border=False):
            if self.project.currentStep == 0:
                self.drawDataset()
            elif self.project.currentStep == 1:
                self.drawModels()
            elif self.project.currentStep == 2:
                drawTrainingForm(self)
            else:
                self.drawReview()
        with dearpygui.group(horizontal=True):
            dearpygui.add_button(label="Back", height=36, enabled=self.project.currentStep > 0, callback=self.previous)
            if self.project.currentStep < 3:
                widgets.primaryButton("Next step", tag="nextStepButton", callback=self.next)
            else:
                widgets.primaryButton("Start training", tag="startTrainingButton", enabled=self.shell.trainingReady(), callback=self.startTraining)
                dearpygui.add_button(label="Training / setup", height=36, callback=lambda: self.shell.showTraining(self.project.id))
            dearpygui.add_button(label="Save now", tag="saveProjectButton", height=36, callback=lambda: self.flush(force=True))
            dearpygui.add_text("Unsaved changes" if self.dirtyAt is not None else "All changes saved", tag="saveState", color=theme.colors.muted)

    def markChanged(self) -> None:
        self.dirtyAt = time.monotonic()
        self.saveFailed = False
        if dearpygui.does_item_exist("saveState"):
            dearpygui.set_value("saveState", "Saving...")
        self.shell.markActivity()

    def flush(self, force: bool = False) -> None:
        self.captionView.detail.capture()
        if self.dirtyAt is None:
            return
        if self.saveFailed and not force:
            raise ValidationError("Changes could not be saved. Use Save now to retry before leaving this project.")
        try:
            self.project = self.shell.workspace.projects.save(self.project)
        except Exception:
            self.saveFailed = True
            if dearpygui.does_item_exist("saveState"):
                dearpygui.set_value("saveState", "Save failed; retry")
            raise
        self.dirtyAt = None
        self.saveFailed = False
        if dearpygui.does_item_exist("saveState"):
            dearpygui.set_value("saveState", "All changes saved")
        self.shell.setStatus("Project saved locally.")

    def tick(self) -> None:
        if time.monotonic() >= self.nextTrainingCheck and dearpygui.does_item_exist("startTrainingButton"):
            self.nextTrainingCheck = time.monotonic() + 0.5
            dearpygui.configure_item("startTrainingButton", enabled=self.shell.trainingReady())
        if self.project.currentStep == 0:
            self.captionView.tick()
        if self.dirtyAt is not None and not self.saveFailed and time.monotonic() - self.dirtyAt >= 0.6:
            self.flush()

    def resize(self) -> None:
        self.captionView.resize()

    def selectStep(self, sender=None, appData=None, userData=0) -> None:
        self.captionView.detail.capture()
        self.project = dataclasses.replace(self.project, currentStep=userData)
        self.markChanged()
        self.flush()
        self.shell.refresh()

    def next(self) -> None:
        self.selectStep(userData=min(3, self.project.currentStep + 1))

    def previous(self) -> None:
        self.selectStep(userData=max(0, self.project.currentStep - 1))

    def drawDataset(self) -> None:
        widgets.pathField(self.shell, "Dataset folder", "datasetPathInput", self.project.datasetPath, True, self.changeDataset)
        self.captionView.draw()
        dearpygui.add_spacer(height=6)
        with dearpygui.collapsing_header(label="Project notes (optional)", tag="projectNotes", default_open=False):
            dearpygui.add_input_text(tag="projectNotesInput", default_value=self.project.notes, multiline=True, height=82, width=-1, hint="Subject, caption conventions, or things to try...", callback=self.changeNotes)

    def changeDataset(self, sender, appData) -> None:
        if appData == self.project.datasetPath:
            return
        self.captionView.detail.capture()
        self.project = dataclasses.replace(self.project, datasetPath=appData)
        self.captionRevision += 1
        self.report = None
        self.markChanged()
        self.captionView.folderChanged()
        self.renderDatasetReport()

    def setCaptions(self, profile) -> None:
        if profile == self.project.captions:
            return
        self.project = self.project.withCaptions(profile)
        self.captionRevision += 1
        self.report = None
        self.markChanged()
        self.renderDatasetReport()

    def changeNotes(self, sender, appData) -> None:
        if len(appData) > 4000:
            dearpygui.set_value(sender, self.project.notes)
            raise ValidationError("Project notes are limited to 4,000 characters.")
        self.project = dataclasses.replace(self.project, notes=appData)
        self.markChanged()

    def scan(self) -> None:
        if self.scanning:
            return
        self.captionView.detail.capture()
        path, profile, revision = self.project.datasetPath, self.project.captions, self.captionRevision
        self.scanning = True
        if dearpygui.does_item_exist("checkDatasetButton"):
            dearpygui.configure_item("checkDatasetButton", enabled=False, label="Checking...")
        self.shell.setStatus("Checking images and the selected caption source in the background...")

        def finish() -> None:
            self.scanning = False
            if self.shell.view is self and not self.shell.isClosing:
                if dearpygui.does_item_exist("checkDatasetButton"):
                    dearpygui.configure_item("checkDatasetButton", enabled=True, label="Recheck dataset" if self.project.currentStep == 3 else "Check dataset")
                self.renderDatasetReport()

        def complete(report) -> None:
            current = self.project.datasetPath == path and self.captionRevision == revision
            if current:
                self.report = report
            finish()
            if self.shell.view is self and not self.shell.isClosing:
                if current:
                    self.shell.setStatus(f"Dataset checked: {report.imageCount} images, {report.captionCount} nonempty captions.")
                    if self.project.currentStep == 0:
                        dearpygui.set_value("datasetCheckDetails", True)
                        self.captionView.refreshImages()
                else:
                    self.shell.setStatus("Dataset or captions changed during the check. Recheck to refresh the report.")

        def failed(error) -> None:
            finish()
            if self.shell.view is self and self.project.datasetPath == path and self.captionRevision == revision:
                self.shell.showError(error)

        self.shell.tasks.submit(lambda: inspectDataset(path, self.shell.tasks.cancelEvent, profile), complete, failed)

    def drawDatasetReport(self) -> None:
        with dearpygui.group(tag="datasetReport"):
            self.drawReportContents()

    def renderDatasetReport(self) -> None:
        if self.shell.view is not self or self.shell.isClosing or not dearpygui.does_item_exist("datasetReport"):
            return
        if dearpygui.does_item_exist("datasetCheckDetails"):
            dearpygui.configure_item("datasetCheckDetails", show=self.report is not None)
        dearpygui.delete_item("datasetReport", children_only=True)
        with dearpygui.group(parent="datasetReport"):
            self.drawReportContents()

    def drawReportContents(self) -> None:
        if self.report is None:
            dearpygui.add_text("Not checked for the current folder and captions. Recheck after changing external files.", color=theme.colors.muted, wrap=0)
            return
        report = self.report
        dearpygui.add_text(f"{report.imageCount} images    /    {report.captionCount} captions    /    {report.missingCaptions} missing or empty", color=theme.colors.accent if report.usable else theme.colors.warning, wrap=0)
        dearpygui.add_text(f"Caption source: {CAPTION_MODES[report.captionMode]}", color=theme.colors.muted)
        for issue in report.issues[:6]:
            dearpygui.add_text(issue, color=theme.colors.warning, wrap=0, bullet=True)
        if len(report.issues) > 6:
            dearpygui.add_text(f"{len(report.issues) - 6} additional issue(s), with details capped at 100.", color=theme.colors.muted)

    def drawModels(self) -> None:
        widgets.section("Choose the three components", "Selections belong to this project. Model files remain shared through the library.")
        self.modelOptions = {}
        for role, label in MODEL_ROLES.items():
            records = self.shell.workspace.models.choices(role)
            options = {"Choose a registered model": ""}
            options.update({f"{record.label} [{record.id[:6]}]": record.id for record in records})
            self.modelOptions[role] = options
            current = self.project.modelIds.get(role, "")
            selected = next((text for text, modelId in options.items() if modelId == current), "Choose a registered model")
            dearpygui.add_text(label)
            dearpygui.add_combo(list(options), default_value=selected, width=-1, callback=self.changeModel, user_data=role, tag=f"{role}Choice")
            record = next((record for record in records if record.id == current), None)
            dearpygui.add_text(record.path if record else "Not selected, or no longer in the library.", tag=f"{role}Path", color=theme.colors.muted, wrap=0)
            dearpygui.add_text(record.fileStatus() if record else "Selection needed", tag=f"{role}Status", color=theme.colors.accent if record and record.fileStatus() == "Registered" else theme.colors.warning)
            dearpygui.add_spacer(height=4)
        dearpygui.add_button(label="Manage model library", height=34, callback=self.manageModels)
        dearpygui.add_text("File registration does not verify architecture or training compatibility.", color=theme.colors.muted, wrap=0)

    def changeModel(self, sender, appData, userData) -> None:
        modelId = self.modelOptions[userData][appData]
        choices = dict(self.project.modelIds)
        if modelId:
            choices[userData] = modelId
        else:
            choices.pop(userData, None)
        self.project = dataclasses.replace(self.project, modelIds=choices)
        self.markChanged()
        record = self.shell.workspace.models.get(modelId) if modelId else None
        dearpygui.set_value(f"{userData}Path", record.path if record else "Not selected.")
        dearpygui.set_value(f"{userData}Status", record.fileStatus() if record else "Selection needed")
        dearpygui.configure_item(f"{userData}Status", color=theme.colors.accent if record and record.fileStatus() == "Registered" else theme.colors.warning)

    def manageModels(self) -> None:
        from animaEasy.ui.modelsView import ModelsView

        self.flush()
        self.shell.navigate(ModelsView(self.shell, returnView=self))

    def changeTraining(self, sender, appData, userData) -> None:
        value = int(appData) if userData == "resolution" else appData
        candidate = dataclasses.replace(self.project.training, **{userData: value})
        if userData in ("epochs", "repeats") and value != getattr(self.project.training, userData):
            candidate = dataclasses.replace(candidate, effortPreset="Custom")
        try:
            candidate.validate()
        except ValidationError:
            dearpygui.set_value(sender, getattr(self.project.training, userData))
            raise
        self.project = dataclasses.replace(self.project, training=candidate)
        self.markChanged()
        self.updatePresetDescription()

    def changePreset(self, sender, appData) -> None:
        self.project = dataclasses.replace(self.project, training=self.project.training.usePreset(appData))
        self.markChanged()
        dearpygui.set_value("trainingEpochs", self.project.training.epochs)
        dearpygui.set_value("trainingRepeats", self.project.training.repeats)
        self.updatePresetDescription()

    def updatePresetDescription(self) -> None:
        if dearpygui.does_item_exist("trainingPreset"):
            dearpygui.set_value("trainingPreset", self.project.training.effortPreset)
            dearpygui.set_value("presetDescription", presetDescription(self.project.training))

    def drawReview(self) -> None:
        widgets.section("Review your training run", "Starting checks CUDA and the input files again before loading the trainer.")
        rows = [("Dataset", self.project.datasetPath or "Not selected"), ("Caption source", CAPTION_MODES[self.project.captions.mode])]
        for role, label in MODEL_ROLES.items():
            try:
                record = self.shell.workspace.models.get(self.project.modelIds.get(role, ""))
                rows.append((label, f"{record.label} — {record.fileStatus()}"))
            except ValidationError:
                rows.append((label, "Selection needed"))
        settings = self.project.training
        rows.extend([("Rank / alpha", f"{settings.rank} / {settings.alpha}"), ("Learning rate", f"{settings.learningRate:g}"), ("Effort preset", settings.effortPreset), ("Epochs / repeats / resolution", f"{settings.epochs} / {settings.repeats} / {settings.resolution} px"), ("Output folders", str(self.shell.workspace.projects.folder(self.project.id) / "runs" / "<run ID>" / "outputs")), ("Optimizer / swapped blocks", f"{settings.optimizer} / {settings.blocksToSwap}"), ("Saved epoch checkpoints", f"Keep {settings.keepCheckpoints}, plus the final weights and trainer state")])
        if self.report is not None and self.report.usable:
            steps = math.ceil(self.report.imageCount * settings.repeats / (settings.batchSize * settings.gradientAccumulation)) * settings.epochs
            rows.append(("Approximate steps", f"{steps:,} — actual bucketing may change this"))
        widgets.details(rows)
        dearpygui.add_spacer(height=8)
        self.drawDatasetReport()
        dearpygui.add_button(label="Checking..." if self.scanning else "Recheck dataset", tag="checkDatasetButton", enabled=not self.scanning, callback=self.scan)
        dearpygui.add_spacer(height=10)
        runtimeMessage = self.shell.training.runtime.probeResult().get("message", "Set up the isolated runtime from Training and pass its CUDA check to enable Start training.")
        dearpygui.add_text(runtimeMessage, color=theme.colors.muted, wrap=0)
        dearpygui.add_text("Captions and settings are captured for each run. Images and model weights stay in place and must remain unchanged. Missing or empty captions train with empty text.", color=theme.colors.muted, wrap=0)
        dearpygui.add_text("GPU capacity alone does not determine the right resolution. Keep a resolution you have tested with these models and settings; 8 GB does not mean a 512 px limit.", color=theme.colors.muted, wrap=0)
        dearpygui.add_text("Resume starts from a fully saved epoch state. Stop or close does not save the current unfinished epoch.", color=theme.colors.muted, wrap=0)


    def startTraining(self) -> None:
        self.flush(force=True)
        project = self.project

        def accepted() -> None:
            self.shell.training.start(project)
            self.shell.showTraining(project.id)

        self.shell.confirm("Start this training run?", "Start one local Anima LoRA job with these settings? The run captures its captions and configuration, while model weights and images stay in place. Keep their original files unchanged. Closing the application stops training; only fully saved epochs can be resumed.", accepted, "Start training")
