from dearpygui import dearpygui

from animaEasy.appearance import ACCENT_OPTIONS, DEFAULT_ACCENT, THEME_MODES
from animaEasy.errors import ValidationError
from animaEasy.ui import theme, widgets


class SettingsView:
    section = "settings"

    def __init__(self, shell):
        self.shell = shell
        self.workspacePath = shell.workspace.preferences.workspacePath

    def draw(self) -> None:
        preferences = self.shell.workspace.preferences
        widgets.pageHeading("PREFERENCES", "Settings", "Make the workspace feel like yours. Appearance changes are saved automatically.")
        with dearpygui.child_window(height=-52, border=False):
            widgets.labelWithHint("Theme", "Auto follows your operating system's app appearance. If the system does not report a preference, Auto uses Dark. Light and Dark stay fixed.", "hintTheme")
            dearpygui.add_combo(list(THEME_MODES), tag="themePreference", default_value=preferences.theme, width=250, callback=self.changeTheme)
            dearpygui.add_text(f"Currently using {self.shell.resolvedTheme.lower()} appearance.", color=theme.colors.muted)
            dearpygui.add_spacer(height=4)
            widgets.labelWithHint("Accent color", "Used for selected items, focus, and primary actions. Pick a color below or open the custom color picker. Very light or dark choices are adjusted for readable text.", "hintAccent")
            selected = next((name for name, color in ACCENT_OPTIONS.items() if color == preferences.accentColor), "Custom")
            with dearpygui.group(horizontal=True):
                dearpygui.add_combo([*ACCENT_OPTIONS, "Custom"], tag="accentPreference", default_value=selected, width=250, callback=self.changeAccent)
                dearpygui.add_color_edit(tag="accentColorInput", default_value=(*preferences.accentColor, 255), width=200, no_alpha=True, display_mode=dearpygui.mvColorEdit_hex, callback=self.customAccent)
                dearpygui.add_button(label="Reset to lapis", callback=self.resetAccent)
            dearpygui.add_spacer(height=12)
            dearpygui.add_separator()
            dearpygui.add_spacer(height=8)
            widgets.section("Workspace", "Changing folders selects another project collection; it does not move or delete projects.")
            widgets.pathField(self.shell, "Project folder", "settingsWorkspacePath", self.workspacePath, True, self.capturePath)
            dearpygui.add_button(label="Use this workspace", height=34, callback=self.saveWorkspace)
            dearpygui.add_spacer(height=12)
            widgets.section("Training runtime")
            dearpygui.add_text(str(self.shell.training.runtime.root), color=theme.colors.muted, wrap=0)
            dearpygui.add_button(label="Change runtime location in Training", callback=self.openRuntimeSettings)
            dearpygui.add_spacer(height=12)
            widgets.section("Application state")
            dearpygui.add_text(str(self.shell.workspace.statePath), color=theme.colors.muted, wrap=0)
            dearpygui.add_text("Preferences and the shared model library are stored here. Input images and caption files are never changed by appearance settings.", color=theme.colors.muted, wrap=0)
        dearpygui.add_text("Everything is stored on this computer.", color=theme.colors.muted)

    def openRuntimeSettings(self) -> None:
        self.shell.showTraining()
        dearpygui.set_value("runtimeSetupPanel", True)

    def capturePath(self, sender, appData) -> None:
        self.workspacePath = appData

    def saveWorkspace(self) -> None:
        if self.shell.training.busy:
            raise ValidationError("Stop the managed training operation before changing workspaces.")
        self.capturePath(None, dearpygui.get_value("settingsWorkspacePath"))
        self.shell.workspace.configure(self.workspacePath)
        self.shell.setStatus("Workspace selected. Existing projects were not moved.")
        self.shell.refresh()

    def changeTheme(self, sender, appData) -> None:
        self.shell.workspace.setAppearance(appData, self.shell.workspace.preferences.accentColor)
        self.shell.applyAppearance()

    def changeAccent(self, sender, appData) -> None:
        if appData in ACCENT_OPTIONS:
            self.shell.workspace.setAppearance(self.shell.workspace.preferences.theme, ACCENT_OPTIONS[appData])
            self.shell.applyAppearance()

    def customAccent(self, sender, appData) -> None:
        accent = tuple(max(0, min(255, round(channel))) for channel in dearpygui.get_value(sender)[:3])
        self.shell.workspace.setAppearance(self.shell.workspace.preferences.theme, accent)
        self.shell.applyAppearance(redraw=False)
        dearpygui.set_value("accentPreference", next((name for name, color in ACCENT_OPTIONS.items() if color == accent), "Custom"))

    def resetAccent(self) -> None:
        self.shell.workspace.setAppearance(self.shell.workspace.preferences.theme, DEFAULT_ACCENT)
        self.shell.applyAppearance()
