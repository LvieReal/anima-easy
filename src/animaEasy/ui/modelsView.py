from dearpygui import dearpygui

from animaEasy.domain import BASE_MODELS, MODEL_ROLES
from animaEasy.models import inspectModel
from animaEasy.ui import theme, widgets


class ModelsView:
    section = "models"

    def __init__(self, shell, setupMode: bool = False, returnView=None):
        self.shell = shell
        self.setupMode = setupMode
        self.returnView = returnView
        self.importing = False
        self.message = ""
        self.role = "diffusion"
        self.variant = BASE_MODELS[0]
        self.label = ""
        self.path = ""

    def draw(self) -> None:
        widgets.pageHeading("FIRST-RUN SETUP  /  2 OF 3" if self.setupMode else "SHARED RESOURCES  /  MODELS", "Reuse the models you have.", "One library for every project. Files are referenced in place, never moved or duplicated.")
        with dearpygui.child_window(height=-68, border=False):
            with dearpygui.table(header_row=False, policy=dearpygui.mvTable_SizingStretchSame):
                dearpygui.add_table_column()
                dearpygui.add_table_column()
                with dearpygui.table_row():
                    with dearpygui.group():
                        dearpygui.add_text("Component")
                        dearpygui.add_combo(list(MODEL_ROLES.values()), default_value=MODEL_ROLES[self.role], width=-1, tag="modelRoleInput", callback=self.changeRole)
                    with dearpygui.group():
                        dearpygui.add_text("Model family")
                        dearpygui.add_combo(self.variants(), default_value=self.variant, width=-1, tag="modelVariantInput", callback=self.capture)
            widgets.pathField(self.shell, "Model file", "modelPathInput", self.path, False, self.capture)
            dearpygui.add_text("Display name (optional)")
            dearpygui.add_input_text(tag="modelLabelInput", default_value=self.label, hint="Uses the filename when left blank", width=-1, callback=self.capture)
            with dearpygui.group(horizontal=True):
                widgets.primaryButton("Checking file..." if self.importing else "Register model", tag="registerModelButton", enabled=not self.importing, callback=self.register)
                dearpygui.add_text("Weights stay on disk.", color=theme.colors.muted)
            dearpygui.add_text("Checks file structure, not model architecture or training compatibility. Select the correct component yourself.", color=theme.colors.muted, wrap=0)
            if self.message:
                dearpygui.add_text(self.message, color=theme.colors.accent, wrap=0)
            dearpygui.add_spacer(height=8)
            dearpygui.add_separator()
            dearpygui.add_spacer(height=6)
            widgets.section("Registered models", "A training project will need a diffusion model, Qwen3 text encoder, and Qwen-Image VAE.")
            self.drawLibrary()
        with dearpygui.group(horizontal=True):
            if self.setupMode:
                dearpygui.add_button(label="Back", height=36, callback=self.back)
                widgets.primaryButton("Continue" if self.shell.workspace.models.records else "Add models later", tag="continueSetupButton", callback=self.next)
            elif self.returnView is not None:
                widgets.primaryButton("Return to project", callback=lambda: self.shell.navigate(self.returnView))
            else:
                dearpygui.add_text("Forgetting an entry never deletes the model file.", color=theme.colors.muted)

    def variants(self) -> list[str]:
        return BASE_MODELS if self.role == "diffusion" else ["Qwen3 0.6B" if self.role == "textEncoder" else "Qwen-Image VAE"]

    def capture(self, sender=None, appData=None) -> None:
        self.path = dearpygui.get_value("modelPathInput")
        self.label = dearpygui.get_value("modelLabelInput")
        self.variant = dearpygui.get_value("modelVariantInput")

    def changeRole(self, sender, appData) -> None:
        self.capture()
        self.role = next(role for role, label in MODEL_ROLES.items() if label == appData)
        self.variant = self.variants()[0]
        dearpygui.configure_item("modelVariantInput", items=self.variants(), default_value=self.variant)

    def register(self) -> None:
        if self.importing:
            return
        self.capture()
        path, role, label, variant = self.path, self.role, self.label, self.variant
        self.importing = True
        dearpygui.configure_item("registerModelButton", enabled=False, label="Checking file...")
        self.shell.setStatus("Inspecting model structure without loading weights...")

        def complete(inspection) -> None:
            self.importing = False
            try:
                record = self.shell.workspace.models.register(inspection, role, label, variant)
            except Exception as error:
                failed(error)
                return
            self.message = f"Registered {record.label}. The original file was not modified."
            self.shell.setStatus(self.message)
            if self.shell.view is self and not self.shell.isClosing:
                self.shell.refresh()

        def failed(error) -> None:
            self.importing = False
            if self.shell.view is self and not self.shell.isClosing:
                self.shell.refresh()
            self.shell.showError(error)

        self.shell.tasks.submit(lambda: inspectModel(path), complete, failed)

    def drawLibrary(self) -> None:
        records = self.shell.workspace.models.records
        if not records:
            with dearpygui.child_window(height=96, border=True):
                dearpygui.add_text("No models registered yet.", color=theme.colors.muted)
                dearpygui.add_text("Use Browse to locate an existing .safetensors file, or add models later.", wrap=0, color=theme.colors.muted)
            return
        with dearpygui.table(header_row=True, policy=dearpygui.mvTable_SizingStretchProp, row_background=True, borders_innerH=True):
            dearpygui.add_table_column(label="Model", init_width_or_weight=3)
            dearpygui.add_table_column(label="Component", init_width_or_weight=1.4)
            dearpygui.add_table_column(label="Status", init_width_or_weight=1.4)
            dearpygui.add_table_column(label="", width_fixed=True, init_width_or_weight=80)
            for record in records:
                with dearpygui.table_row():
                    with dearpygui.group():
                        dearpygui.add_text(record.label, wrap=0)
                        item = dearpygui.add_text(widgets.readablePath(record.path, 40), color=theme.colors.muted)
                        with dearpygui.tooltip(item):
                            dearpygui.add_text(record.path, wrap=600)
                        theme.useFont(item, "captionFont")
                    dearpygui.add_text(MODEL_ROLES[record.role], wrap=0)
                    with dearpygui.group():
                        status = record.fileStatus()
                        dearpygui.add_text(status, color=theme.colors.accent if status == "Registered" else theme.colors.warning, wrap=0)
                        dearpygui.add_text(widgets.formatSize(record.sizeBytes), color=theme.colors.muted)
                    dearpygui.add_button(label="Forget", callback=self.askForget, user_data=record.id)

    def askForget(self, sender, appData, userData) -> None:
        record = self.shell.workspace.models.get(userData)
        self.shell.confirm("Forget model?", f"Remove {record.label} from the library? The file stays on disk. Projects using this entry will need another model selection.", lambda: self.forget(record.id))

    def forget(self, modelId: str) -> None:
        self.capture()
        self.shell.workspace.models.forget(modelId)
        self.shell.setStatus("Library entry removed. The model file was not deleted.")
        self.shell.refresh()

    def back(self) -> None:
        from animaEasy.ui.setupView import SetupView

        self.shell.navigate(SetupView(self.shell))

    def next(self) -> None:
        from animaEasy.ui.setupView import SetupView

        self.shell.navigate(SetupView(self.shell, step=2))
