import logging
import time

from dearpygui import dearpygui

from animaEasy import VERSION
from animaEasy.appearance import resolveTheme, systemTheme
from animaEasy.errors import AnimaError, StorageError
from animaEasy.tasks import TaskQueue
from animaEasy.training import TrainingController
from animaEasy.ui import theme, widgets
from animaEasy.ui.editorView import EditorView
from animaEasy.ui.modelsView import ModelsView
from animaEasy.ui.projectsView import ProjectsView
from animaEasy.ui.settingsView import SettingsView
from animaEasy.ui.setupView import SetupView
from animaEasy.ui.trainingView import TrainingView


SIDEBAR_WIDTH = 204
WINDOW_PADDING = 16
ITEM_SPACING = 10


class Desktop:
    def __init__(self, workspace):
        self.workspace = workspace
        self.tasks = TaskQueue()
        self.training = TrainingController(workspace)
        self.nextRuntimeCheck = 0
        self.view = None
        self.lastActivity = time.monotonic()
        self.lastMouse = None
        self.isClosing = False
        self.started = False
        self.themePreference = workspace.preferences.theme
        self.resolvedTheme = resolveTheme(self.themePreference)
        self.nextThemeCheck = time.monotonic() + 30
        self.themeCheckPending = False
        dearpygui.create_context()
        dearpygui.configure_app(manual_callback_management=True)
        theme.installTheme(self.resolvedTheme, workspace.preferences.accentColor)
        theme.installFonts()
        dearpygui.add_texture_registry(tag="imageTextures")
        dearpygui.create_viewport(title="Anima Easy", width=1240, height=850, min_width=1024, min_height=700, vsync=False, clear_color=theme.colors.background)
        self.buildShell()
        dearpygui.set_viewport_resize_callback(self.resize)
        with dearpygui.handler_registry():
            dearpygui.add_mouse_click_handler(callback=self.markActivity)
            dearpygui.add_mouse_wheel_handler(callback=self.markActivity)
            dearpygui.add_key_press_handler(callback=self.markActivity)
            dearpygui.add_key_press_handler(key=dearpygui.mvKey_S, callback=self.keyboardSave)
        self.navigate(ProjectsView(self) if workspace.preferences.setupComplete else SetupView(self))

    def buildShell(self) -> None:
        with dearpygui.window(tag="rootWindow", no_title_bar=True, no_move=True, no_resize=True, no_scrollbar=True):
            with dearpygui.group(horizontal=True):
                with dearpygui.child_window(tag="sidebar", width=SIDEBAR_WIDTH, height=780, border=False, always_use_window_padding=True):
                    theme.useFont(dearpygui.add_text("ANIMA EASY", tag="brandLabel", color=theme.colors.text), "brandFont")
                    theme.useFont(dearpygui.add_text("LOCAL WORKSPACE", tag="brandSubtitle", color=theme.colors.muted), "captionFont")
                    dearpygui.add_spacer(height=24)
                    theme.useFont(dearpygui.add_text("WORKSPACE", tag="navigationLabel", color=theme.colors.muted), "captionFont")
                    dearpygui.add_button(label="Projects", tag="navProjects", width=-1, height=38, callback=self.showProjects)
                    dearpygui.add_button(label="Model library", tag="navModels", width=-1, height=38, callback=lambda: self.navigate(ModelsView(self)))
                    dearpygui.add_button(label="Training", tag="navTraining", width=-1, height=38, callback=lambda: self.showTraining())
                    dearpygui.add_button(label="Settings", tag="navWorkspace", width=-1, height=38, callback=lambda: self.navigate(SettingsView(self)))
                    dearpygui.add_button(label="Finish setup", tag="navSetup", width=-1, height=34, callback=lambda: self.navigate(SetupView(self)))
                    dearpygui.add_spacer(tag="sidebarSpacer", height=280)
                    dearpygui.add_separator()
                    theme.useFont(dearpygui.add_text("LOCAL DESKTOP", tag="desktopLabel", color=theme.colors.accent), "captionFont")
                    dearpygui.add_text(f"Version {VERSION}", tag="versionLabel", color=theme.colors.muted)
                    dearpygui.add_text("Training runtime\nnot installed", tag="runtimeLabel", color=theme.colors.muted)
                dearpygui.bind_item_theme("sidebar", "sidebarTheme")
                dearpygui.add_child_window(tag="pageHost", width=986, height=780, border=False, always_use_window_padding=True)
                dearpygui.bind_item_theme("pageHost", "pageTheme")
            theme.useFont(dearpygui.add_text("Ready. Your files stay on this computer.", tag="statusLine", color=theme.colors.muted), "captionFont")

    def start(self) -> None:
        dearpygui.setup_dearpygui()
        dearpygui.show_viewport()
        dearpygui.set_primary_window("rootWindow", True)
        self.started = True
        self.resize()

    def navigate(self, view) -> None:
        if self.view is not None and hasattr(self.view, "flush"):
            self.view.flush()
        self.view = view
        self.refresh()
        self.markActivity()

    def releasePreviewTexture(self) -> None:
        if dearpygui.does_item_exist("captionPreviewTexture"):
            dearpygui.delete_item("captionPreviewTexture")

    def refresh(self) -> None:
        if self.isClosing:
            return
        dearpygui.delete_item("pageHost", children_only=True)
        self.releasePreviewTexture()
        try:
            with dearpygui.group(parent="pageHost"):
                self.view.draw()
        except (AnimaError, OSError) as error:
            dearpygui.delete_item("pageHost", children_only=True)
            self.releasePreviewTexture()
            with dearpygui.group(parent="pageHost"):
                widgets.pageHeading("WORKSPACE", "This page could not be opened.", "Your files have not been reset. Use Settings to choose another workspace folder.")
                dearpygui.add_text(str(error), color=theme.colors.error, wrap=0)
            self.setStatus(str(error))
        for section, tag in (("projects", "navProjects"), ("models", "navModels"), ("training", "navTraining"), ("settings", "navWorkspace"), ("setup", "navSetup")):
            dearpygui.bind_item_theme(tag, "selectedTheme" if self.view.section == section else 0)
        dearpygui.configure_item("navSetup", show=not self.workspace.preferences.setupComplete)
        if self.started:
            self.resize()

    def applyAppearance(self, redraw: bool = True, resolvedMode: str | None = None) -> None:
        preferences = self.workspace.preferences
        if resolvedMode is not None:
            self.resolvedTheme = resolvedMode
        elif preferences.theme != self.themePreference:
            self.resolvedTheme = resolveTheme(preferences.theme)
        self.themePreference = preferences.theme
        theme.updateTheme(self.resolvedTheme, preferences.accentColor)
        for tag, color in (("brandLabel", theme.colors.text), ("brandSubtitle", theme.colors.muted), ("navigationLabel", theme.colors.muted), ("desktopLabel", theme.colors.accent), ("versionLabel", theme.colors.muted), ("runtimeLabel", theme.colors.muted), ("statusLine", theme.colors.muted), ("pageKicker", theme.colors.accent)):
            if dearpygui.does_item_exist(tag):
                dearpygui.configure_item(tag, color=color)
        if redraw:
            self.refresh()
        self.markActivity()

    def checkSystemTheme(self) -> None:
        if self.workspace.preferences.theme != "Auto" or self.themeCheckPending or time.monotonic() < self.nextThemeCheck:
            return
        self.themeCheckPending = True
        self.nextThemeCheck = time.monotonic() + 30

        def complete(mode) -> None:
            self.themeCheckPending = False
            if not self.isClosing and self.workspace.preferences.theme == "Auto" and mode != self.resolvedTheme:
                self.applyAppearance(resolvedMode=mode)

        def failed(error) -> None:
            self.themeCheckPending = False
            logging.debug("System appearance could not be checked: %s", error)

        self.tasks.submit(systemTheme, complete, failed)

    def showTraining(self, projectId: str | None = None) -> None:
        self.navigate(TrainingView(self, projectId))

    def trainingReady(self) -> bool:
        return not self.training.busy and self.training.runtime.probeResult().get("cudaReady", False)

    def showProjects(self) -> None:
        self.navigate(ProjectsView(self))

    def openProject(self, projectId: str) -> None:
        if self.view is not None and hasattr(self.view, "flush"):
            self.view.flush()
        project = self.workspace.projects.load(projectId)
        self.navigate(EditorView(self, project))

    def setStatus(self, message: str) -> None:
        logging.info(message)
        if not self.isClosing:
            dearpygui.set_value("statusLine", widgets.readablePath(message.replace("\n", " "), 145))

    def showError(self, error: Exception) -> None:
        logging.error("Application operation failed", exc_info=(type(error), error, error.__traceback__))
        if self.isClosing:
            return
        self.setStatus(str(error))
        self.closeDialog("errorDialog")
        with dearpygui.window(label="Action could not be completed", tag="errorDialog", modal=True, no_resize=True, width=650, pos=self.dialogPosition(650, 240)):
            dearpygui.add_text(str(error), color=theme.colors.error, wrap=600)
            dearpygui.add_text("Existing files have not been reset or replaced with defaults.", color=theme.colors.muted, wrap=600)
            dearpygui.add_button(label="Close", width=90, height=34, callback=lambda: self.closeDialog("errorDialog"))

    def confirm(self, title: str, message: str, action, confirmLabel: str = "Forget entry") -> None:
        self.closeDialog("confirmDialog")

        def accepted() -> None:
            self.closeDialog("confirmDialog")
            action()

        with dearpygui.window(label=title, tag="confirmDialog", modal=True, no_resize=True, width=590, pos=self.dialogPosition(590, 220)):
            dearpygui.add_text(message, wrap=540)
            with dearpygui.group(horizontal=True):
                dearpygui.add_button(label="Cancel", tag="cancelConfirmation", width=90, height=34, callback=lambda: self.closeDialog("confirmDialog"))
                widgets.primaryButton(confirmLabel, tag="acceptConfirmation", callback=accepted)

    def closeDialog(self, tag: str) -> None:
        if dearpygui.does_item_exist(tag):
            dearpygui.delete_item(tag)

    def browse(self, target: str, directory: bool, onChange=None, fileExtensions: tuple[str, ...] = (".safetensors",), userData=None, browseFolder: str = "") -> None:
        self.closeDialog("pathDialog")
        defaultPath = widgets.defaultBrowsePath(dearpygui.get_value(target), browseFolder)
        originView = self.view

        def selected(sender, appData) -> None:
            self.closeDialog("pathDialog")
            if self.view is not originView or not dearpygui.does_item_exist(target):
                return
            path = widgets.selectedDialogPath(appData, directory)
            if path:
                dearpygui.set_value(target, path)
                if onChange is not None:
                    dearpygui.run_callbacks([[onChange, target, path, userData]])

        title = "Choose a folder" if directory else ("Choose a caption file" if fileExtensions == (".txt",) else "Choose a model")
        with dearpygui.file_dialog(tag="pathDialog", label=title, directory_selector=directory, default_path=defaultPath, default_filename="." if directory else "", file_count=1, width=840, height=510, modal=True, callback=selected, cancel_callback=lambda: self.closeDialog("pathDialog")):
            if not directory:
                for extension in fileExtensions:
                    dearpygui.add_file_extension(extension, color=theme.colors.accent)
            dearpygui.add_file_extension(".*")

    def dialogPosition(self, width: int, height: int) -> list[int]:
        return [max(20, (dearpygui.get_viewport_client_width() - width) // 2), max(20, (dearpygui.get_viewport_client_height() - height) // 2)]

    def pageWidth(self) -> int:
        return max(650, dearpygui.get_viewport_client_width() - 2 * WINDOW_PADDING - SIDEBAR_WIDTH - ITEM_SPACING)

    def stepWidth(self) -> int:
        return max(130, (self.pageWidth() - 2 * WINDOW_PADDING - 3 * ITEM_SPACING) // 4)

    def resize(self, sender=None, appData=None) -> None:
        height = dearpygui.get_viewport_client_height()
        dearpygui.configure_item("sidebar", height=max(300, height - 70))
        dearpygui.configure_item("pageHost", width=self.pageWidth(), height=max(300, height - 70))
        dearpygui.configure_item("sidebarSpacer", height=max(50, height - (613 if not self.workspace.preferences.setupComplete else 568)))
        for index in range(4):
            if dearpygui.does_item_exist(f"stepButton{index}"):
                dearpygui.set_item_width(f"stepButton{index}", self.stepWidth())
        if hasattr(self.view, "resize"):
            self.view.resize()
        self.markActivity()

    def markActivity(self, sender=None, appData=None) -> None:
        self.lastActivity = time.monotonic()

    def keyboardSave(self, sender=None, appData=None) -> None:
        controlDown = dearpygui.is_key_down(dearpygui.mvKey_LControl) or dearpygui.is_key_down(dearpygui.mvKey_RControl)
        if controlDown and hasattr(self.view, "flush"):
            self.view.flush(force=True)

    def processEvents(self) -> None:
        for job in dearpygui.get_callback_queue() or []:
            try:
                dearpygui.run_callbacks([job])
            except Exception as error:
                self.showError(error)
        try:
            self.tasks.poll()
            if time.monotonic() >= self.nextRuntimeCheck:
                self.nextRuntimeCheck = time.monotonic() + 0.5
                installed = self.training.runtime.installed()
                status = "Job in progress" if self.training.busy else ("Installed" if installed else "Not installed")
                dearpygui.set_value("runtimeLabel", f"Training runtime\n{status}")
            if hasattr(self.view, "tick"):
                self.view.tick()
            self.checkSystemTheme()
        except Exception as error:
            self.showError(error)

    def frame(self) -> None:
        self.processEvents()
        dearpygui.render_dearpygui_frame()

    def run(self) -> None:
        if not self.started:
            self.start()
        while dearpygui.is_dearpygui_running():
            startedAt = time.monotonic()
            self.frame()
            mouse = dearpygui.get_mouse_pos(local=False)
            if mouse != self.lastMouse:
                self.lastMouse = mouse
                if 0 <= mouse[0] < dearpygui.get_viewport_client_width() and 0 <= mouse[1] < dearpygui.get_viewport_client_height():
                    self.markActivity()
            framesPerSecond = 30 if startedAt - self.lastActivity < 1.5 else 8
            time.sleep(max(0, 1 / framesPerSecond - (time.monotonic() - startedAt)))

    def close(self) -> None:
        try:
            self.processEvents()
            if self.view is not None and hasattr(self.view, "flush"):
                self.view.flush(force=True)
        except Exception as error:
            if isinstance(self.view, EditorView) and self.view.dirtyAt is not None:
                recoveryPath = self.workspace.preserveDraft(self.view.project)
                raise StorageError(f"The project could not be saved. A recovery draft was preserved at {recoveryPath}. Original error: {error}") from error
            raise
        finally:
            self.isClosing = True
            try:
                self.training.close()
            finally:
                try:
                    self.tasks.close()
                finally:
                    dearpygui.destroy_context()


def showStartupError(message: str) -> None:
    dearpygui.create_context()
    try:
        theme.installTheme()
        theme.installFonts()
        dearpygui.create_viewport(title="Anima Easy — startup issue", width=760, height=350, min_width=600, min_height=280, vsync=False)
        with dearpygui.window(tag="startupWindow"):
            theme.useFont(dearpygui.add_text("Unable to open the workspace"), "brandFont")
            dearpygui.add_text(message, wrap=700, color=theme.colors.error)
            dearpygui.add_text("No settings or projects were reset. Check the path or repair the named file, then reopen Anima Easy.", wrap=700, color=theme.colors.muted)
            dearpygui.add_button(label="Close", width=100, callback=dearpygui.stop_dearpygui)
        dearpygui.setup_dearpygui()
        dearpygui.show_viewport()
        dearpygui.set_primary_window("startupWindow", True)
        while dearpygui.is_dearpygui_running():
            dearpygui.render_dearpygui_frame()
            time.sleep(0.05)
    finally:
        dearpygui.destroy_context()
