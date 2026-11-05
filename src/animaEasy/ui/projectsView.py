import datetime

from dearpygui import dearpygui

from animaEasy.ui import theme, widgets


STEPS = ("Dataset", "Models", "Training settings", "Review")


class ProjectsView:
    section = "projects"

    def __init__(self, shell):
        self.shell = shell

    def draw(self) -> None:
        listing = self.shell.workspace.projects.list()
        widgets.pageHeading("WORKSPACE  /  PROJECTS", "Your projects", "A focused place for each dataset, model selection, and training configuration.")
        with dearpygui.group(horizontal=True):
            widgets.primaryButton("+  New project", tag="newProjectButton", callback=self.newProject)
            dearpygui.add_text(f"{len(listing.projects)} saved project{'s' if len(listing.projects) != 1 else ''}", color=theme.colors.muted)
        dearpygui.add_spacer(height=8)
        with dearpygui.child_window(height=-54, border=False):
            for error in listing.errors:
                dearpygui.add_text(f"Could not open a project: {error}", color=theme.colors.warning, wrap=0)
            if not listing.projects:
                with dearpygui.child_window(height=215, border=True):
                    dearpygui.add_spacer(height=12)
                    theme.useFont(dearpygui.add_text("Start with one project."), "brandFont")
                    dearpygui.add_text("Name it, choose a dataset, and work through four short steps.\nYour settings are saved locally as you go.", color=theme.colors.muted, wrap=0)
                    dearpygui.add_spacer(height=12)
                    dearpygui.add_text("01  Dataset    /    02  Models    /    03  Settings    /    04  Review", color=theme.colors.accent, wrap=0)
            else:
                with dearpygui.table(header_row=True, row_background=True, borders_innerH=True, policy=dearpygui.mvTable_SizingStretchProp):
                    dearpygui.add_table_column(label="Project", init_width_or_weight=2.5)
                    dearpygui.add_table_column(label="Draft step", init_width_or_weight=1.4)
                    dearpygui.add_table_column(label="Last saved", init_width_or_weight=1.4)
                    for project in listing.projects:
                        with dearpygui.table_row():
                            dearpygui.add_button(label=project.name, width=-1, height=38, callback=self.openProject, user_data=project.id)
                            dearpygui.add_text(STEPS[project.currentStep])
                            localTime = datetime.datetime.fromisoformat(project.updatedAt).astimezone()
                            dearpygui.add_text(localTime.strftime("%b %d, %Y  %H:%M"), color=theme.colors.muted)
            dearpygui.add_spacer(height=18)
            dearpygui.add_text("Projects keep model choices, captions, and training settings together. Start a managed training run from Review.", color=theme.colors.muted, wrap=0)
        dearpygui.add_text(widgets.readablePath(self.shell.workspace.preferences.workspacePath), color=theme.colors.muted)

    def newProject(self) -> None:
        self.shell.closeDialog("projectDialog")
        with dearpygui.window(label="New project", tag="projectDialog", modal=True, no_resize=True, width=500, pos=self.shell.dialogPosition(500, 230), on_close=lambda: self.shell.closeDialog("projectDialog")):
            dearpygui.add_text("Give this training project a name.", color=theme.colors.muted)
            dearpygui.add_input_text(tag="projectNameInput", width=-1, hint="e.g. Character study", on_enter=True, callback=self.createProject)
            dearpygui.add_text("Names do not become folder paths. Each project has a unique ID.", color=theme.colors.muted, wrap=450)
            with dearpygui.group(horizontal=True):
                widgets.primaryButton("Create project", tag="createProjectButton", callback=self.createProject)
                dearpygui.add_button(label="Cancel", height=36, callback=lambda: self.shell.closeDialog("projectDialog"))
        dearpygui.focus_item("projectNameInput")

    def createProject(self, sender=None, appData=None) -> None:
        project = self.shell.workspace.projects.create(dearpygui.get_value("projectNameInput"))
        self.shell.closeDialog("projectDialog")
        self.shell.openProject(project.id)
        self.shell.setStatus("Project created. Changes are saved automatically.")

    def openProject(self, sender, appData, userData) -> None:
        self.shell.openProject(userData)
