from dearpygui import dearpygui

from animaEasy.domain import MODEL_ROLES
from animaEasy.errors import ValidationError
from animaEasy.ui import theme, widgets


class SetupView:
    section = "setup"

    def __init__(self, shell, step: int = 0):
        self.shell = shell
        self.step = step

    def draw(self) -> None:
        workspace = self.shell.workspace
        if self.step == 2:
            self.drawFinish()
            return
        configured = workspace.preferences.setupComplete
        widgets.pageHeading("WORKSPACE" if configured else "FIRST-RUN SETUP  /  1 OF 3", "Workspace settings" if configured else "A place for every project.", "Choose where your projects live. Your existing models and datasets stay where they are.")
        with dearpygui.child_window(height=-68, border=False):
            widgets.section("Project storage", "Each project keeps its own model choices, captions, and training settings.")
            widgets.pathField(self.shell, "Workspace folder", "workspacePathInput", workspace.preferences.workspacePath, True)
            dearpygui.add_spacer(height=10)
            with dearpygui.child_window(height=170, border=True):
                dearpygui.add_text("Keep the heavy files in place", color=theme.colors.accent)
                dearpygui.add_text("Register models by path, including files from an existing ComfyUI installation. Files are referenced in place, without copying or downloading.", wrap=0, color=theme.colors.muted)
                dearpygui.add_text("Changing workspaces does not move or delete existing projects.", wrap=0, color=theme.colors.muted)
            dearpygui.add_spacer(height=12)
            widgets.section("Application state")
            dearpygui.add_text(workspace.statePath.as_posix(), color=theme.colors.muted, wrap=0)
            dearpygui.add_spacer(height=10)
            dearpygui.add_text("This step stores paths and preferences. Training runtime setup is a separate, explicit action on the Training page.", color=theme.colors.warning, wrap=0)
        with dearpygui.group(horizontal=True):
            widgets.primaryButton("Use workspace" if configured else "Continue to models", tag="saveWorkspaceButton", callback=self.save)
            dearpygui.add_text("Local files only. No account required.", color=theme.colors.muted)

    def save(self) -> None:
        if self.shell.training.busy:
            raise ValidationError("Stop the managed training operation before changing workspaces.")
        from animaEasy.ui.modelsView import ModelsView

        self.shell.workspace.configure(dearpygui.get_value("workspacePathInput"))
        self.shell.setStatus("Workspace saved. Existing files were not moved.")
        if not self.shell.workspace.preferences.setupComplete:
            self.shell.navigate(ModelsView(self.shell, setupMode=True))
        else:
            self.shell.refresh()

    def drawFinish(self) -> None:
        workspace = self.shell.workspace
        widgets.pageHeading("FIRST-RUN SETUP  /  3 OF 3", "Your workspace is ready.", "Create a project now. You can return to the model library at any time.")
        with dearpygui.child_window(height=-68, border=False):
            rows = [("Workspace", workspace.preferences.workspacePath)]
            for role, label in MODEL_ROLES.items():
                count = len(workspace.models.choices(role))
                rows.append((label, f"{count} registered" if count else "Not added yet"))
            widgets.details(rows)
            dearpygui.add_spacer(height=16)
            widgets.section("Your workspace")
            for text in ("Create separate projects and reopen them later.", "Register existing model files without copying them.", "Manage captions, check datasets, and save training settings."):
                dearpygui.add_text(text, bullet=True, wrap=0)
            dearpygui.add_spacer(height=12)
            dearpygui.add_text("Open Training when you are ready to install the isolated runtime and check your NVIDIA GPU.", color=theme.colors.warning, wrap=0)
        with dearpygui.group(horizontal=True):
            dearpygui.add_button(label="Back", height=36, callback=self.back)
            widgets.primaryButton("Open projects", tag="finishSetupButton", callback=self.finish)

    def back(self) -> None:
        from animaEasy.ui.modelsView import ModelsView

        self.shell.navigate(ModelsView(self.shell, setupMode=True))

    def finish(self) -> None:
        if self.shell.training.busy:
            raise ValidationError("Stop the managed training operation before changing workspaces.")
        self.shell.workspace.configure(self.shell.workspace.preferences.workspacePath, setupComplete=True)
        self.shell.showProjects()
