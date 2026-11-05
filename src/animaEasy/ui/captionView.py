import dataclasses
import time

from dearpygui import dearpygui

from animaEasy.captions import CAPTION_MODES, CaptionCheck, checkCaptions, datasetScope
from animaEasy.errors import ValidationError
from animaEasy.images import listImages
from animaEasy.ui import theme, widgets
from animaEasy.ui.captionDetail import CaptionDetail


PAGE_SIZE = 100


class CaptionView:
    def __init__(self, editor):
        self.editor = editor
        self.shell = editor.shell
        self.scope = datasetScope(editor.project.datasetPath)
        self.index = None
        self.checks = {}
        self.selected = set()
        self.activeName = ""
        self.search = ""
        self.page = 0
        self.visibleNames = []
        self.rowTags = {}
        self.loading = False
        self.loadGeneration = 0
        self.loadFuture = None
        self.lastAttemptScope = None
        self.changedAt = 0.0
        self.message = "Choose a folder to see its images and captions."
        self.detail = CaptionDetail(self)

    @property
    def profile(self):
        return self.editor.project.captions

    @property
    def visible(self) -> bool:
        return not self.shell.isClosing and self.shell.view is self.editor and self.editor.project.currentStep == 0 and dearpygui.does_item_exist("captionSection")

    def tick(self) -> None:
        if self.scope and not self.loading and self.lastAttemptScope != self.scope and time.monotonic() - self.changedAt > 0.5:
            self.refreshImages()
        if self.visible:
            self.detail.tick()

    def folderChanged(self) -> None:
        self.scope = datasetScope(self.editor.project.datasetPath)
        self.index = None
        self.checks = {}
        self.selected.clear()
        self.activeName = ""
        self.search = ""
        self.page = 0
        self.loading = False
        self.loadGeneration += 1
        if self.loadFuture is not None:
            self.loadFuture.cancel()
        self.lastAttemptScope = None
        self.changedAt = time.monotonic()
        self.message = "Reading this folder..." if self.scope else "Choose a folder to see its images and captions."
        self.detail.reset()
        self.render()

    def draw(self) -> None:
        with dearpygui.group(tag="captionSection"):
            self.drawContents()

    def render(self) -> None:
        if not self.visible:
            return
        dearpygui.delete_item("captionSection", children_only=True)
        self.shell.releasePreviewTexture()
        with dearpygui.group(parent="captionSection"):
            self.drawContents()

    def drawContents(self) -> None:
        widgets.labelWithHint("Caption source", "Matching files are read-only. Assigned files can be shared by multiple images. Project text is editable here. Switching modes preserves each mode's captions and assignments. Each folder path keeps its own caption profile.", "hintCaptionSource")
        with dearpygui.group(horizontal=True):
            dearpygui.add_combo(list(CAPTION_MODES.values()), tag="captionMode", default_value=CAPTION_MODES[self.profile.mode], width=300, enabled=bool(self.scope), callback=self.changeMode)
            dearpygui.add_button(label="Loading..." if self.loading else "Refresh images", tag="refreshImagesButton", enabled=bool(self.scope) and not self.loading, callback=self.refreshImages)
            dearpygui.add_button(label="Checking..." if self.editor.scanning else "Check dataset", tag="checkDatasetButton", height=26, enabled=not self.editor.scanning, callback=self.editor.scan)
        with dearpygui.collapsing_header(label="Dataset check", tag="datasetCheckDetails", default_open=False, show=self.editor.report is not None):
            self.editor.drawDatasetReport()
        if self.index is None or not self.index.names:
            dearpygui.add_text(self.message if self.index is None else "No supported images found. Subfolders are not included.", color=theme.colors.muted, wrap=0)
            return
        height = self.galleryHeight()
        with dearpygui.group(horizontal=True):
            with dearpygui.child_window(tag="captionBrowser", width=self.listWidth(), height=height, border=True, no_scrollbar=True):
                dearpygui.add_input_text(tag="captionFilter", default_value=self.search, hint="Filter image names...", width=-1, callback=self.filterNames)
                with dearpygui.group(horizontal=True):
                    dearpygui.add_button(label="Select page", tag="selectCaptionPage", callback=self.selectPage)
                    dearpygui.add_button(label="Clear", tag="clearCaptionSelection", callback=self.clearSelection)
                    widgets.hint("Click an image name to edit it. Checkboxes choose bulk targets; selection is retained across pages and filters. Apply to all includes every image in the folder, even images hidden by a filter.", "hintCaptionSelection")
                dearpygui.add_child_window(tag="captionImageList", height=-36, border=False)
                with dearpygui.group(horizontal=True):
                    dearpygui.add_button(label="<", tag="captionPreviousPage", callback=self.previousPage, width=28)
                    dearpygui.add_text("", tag="captionPageLabel")
                    dearpygui.add_button(label=">", tag="captionNextPage", callback=self.nextPage, width=28)
            with dearpygui.child_window(tag="captionDetail", width=-1, height=height, border=True):
                self.detail.draw()
        self.drawRows()
        self.updateSelection()

    def galleryHeight(self) -> int:
        return max(190, min(380, dearpygui.get_viewport_client_height() - 508))

    def listWidth(self) -> int:
        return max(250, min(320, int((dearpygui.get_viewport_client_width() - 278) * 0.34)))

    def resize(self) -> None:
        if not self.visible or not dearpygui.does_item_exist("captionBrowser"):
            return
        height = self.galleryHeight()
        dearpygui.configure_item("captionBrowser", width=self.listWidth(), height=height)
        dearpygui.configure_item("captionDetail", height=height)
        self.detail.resize()

    def drawRows(self) -> None:
        if not self.visible or not dearpygui.does_item_exist("captionImageList"):
            return
        names = [name for name in self.index.names if self.search.casefold() in name.casefold()]
        lastPage = max(0, (len(names) - 1) // PAGE_SIZE)
        self.page = min(self.page, lastPage)
        self.visibleNames = names[self.page * PAGE_SIZE:(self.page + 1) * PAGE_SIZE]
        dearpygui.delete_item("captionImageList", children_only=True)
        self.rowTags = {}
        with dearpygui.table(parent="captionImageList", header_row=False, policy=dearpygui.mvTable_SizingStretchProp):
            dearpygui.add_table_column(width_fixed=True, init_width_or_weight=20)
            dearpygui.add_table_column()
            dearpygui.add_table_column(width_fixed=True, init_width_or_weight=64)
            for offset, name in enumerate(self.visibleNames):
                checkTag, nameTag, stateTag = f"captionCheck{offset}", f"captionImage{offset}", f"captionState{offset}"
                self.rowTags[name] = (checkTag, nameTag, stateTag)
                with dearpygui.table_row():
                    dearpygui.add_checkbox(tag=checkTag, default_value=name in self.selected, callback=self.toggleSelected, user_data=(self.scope, name))
                    dearpygui.add_selectable(tag=nameTag, label=name, default_value=name == self.activeName, callback=self.selectImage, user_data=(self.scope, name))
                    with dearpygui.tooltip(nameTag):
                        dearpygui.add_text(name, wrap=320)
                    state = self.checks.get(name, CaptionCheck("Unchecked"))
                    dearpygui.add_text(state.status, tag=stateTag, color=theme.colors.muted if state.status == "Ready" else theme.colors.warning)
                    with dearpygui.tooltip(stateTag, tag=f"{stateTag}Tooltip", show=bool(state.issue)):
                        dearpygui.add_text(state.issue, tag=f"{stateTag}Issue", wrap=320)
        dearpygui.set_value("captionPageLabel", f"{self.page + 1}/{lastPage + 1} · {len(names)} images")
        dearpygui.configure_item("captionPreviousPage", enabled=self.page > 0)
        dearpygui.configure_item("captionNextPage", enabled=self.page < lastPage)

    def filterNames(self, sender, appData) -> None:
        self.search = appData
        self.page = 0
        self.drawRows()

    def previousPage(self) -> None:
        self.page = max(0, self.page - 1)
        self.drawRows()

    def nextPage(self) -> None:
        self.page += 1
        self.drawRows()

    def toggleSelected(self, sender, appData, userData) -> None:
        scope, name = userData
        if scope != self.scope or self.index is None or name not in self.index.names:
            return
        if appData:
            self.selected.add(name)
        else:
            self.selected.discard(name)
        self.updateSelection()

    def selectPage(self) -> None:
        self.selected.update(self.visibleNames)
        self.updateSelection()

    def clearSelection(self) -> None:
        self.selected.clear()
        self.updateSelection()

    def updateSelection(self) -> None:
        if not self.visible:
            return
        for name, (checkTag, nameTag, _stateTag) in self.rowTags.items():
            dearpygui.set_value(checkTag, name in self.selected)
            dearpygui.set_value(nameTag, name == self.activeName)
        if dearpygui.does_item_exist("applyCaptionSelected"):
            dearpygui.configure_item("applyCaptionSelected", label=f"To selected ({len(self.selected)})", enabled=bool(self.selected))

    def selectImage(self, sender, appData, userData) -> None:
        scope, name = userData
        if scope != self.scope or self.index is None or name not in self.index.names:
            return
        self.detail.capture()
        self.activeName = name
        self.detail.render()
        self.updateSelection()

    def changeMode(self, sender, appData) -> None:
        if not self.visible:
            return
        self.detail.capture()
        mode = next(key for key, label in CAPTION_MODES.items() if label == appData)
        self.editor.setCaptions(dataclasses.replace(self.profile, mode=mode))
        self.checks = checkCaptions(self.index.folder, self.index.names, self.profile) if self.index is not None and mode == "inline" else {}
        self.detail.resetCaption()
        self.render()
        self.refreshImages()

    def updateCheck(self, name: str, state: CaptionCheck) -> None:
        self.checks[name] = state
        if self.visible and name in self.rowTags:
            stateTag = self.rowTags[name][2]
            dearpygui.set_value(stateTag, state.status)
            dearpygui.set_value(f"{stateTag}Issue", state.issue)
            dearpygui.configure_item(f"{stateTag}Tooltip", show=bool(state.issue))
            dearpygui.configure_item(stateTag, color=theme.colors.muted if state.status == "Ready" else theme.colors.warning)

    def refreshImages(self) -> None:
        if not self.scope:
            return
        self.detail.capture()
        self.loadGeneration += 1
        generation, scope, revision = self.loadGeneration, self.scope, self.editor.captionRevision
        path, profile = self.editor.project.datasetPath, self.profile
        if self.loadFuture is not None:
            self.loadFuture.cancel()
        self.loading = True
        self.lastAttemptScope = scope
        if self.visible:
            dearpygui.configure_item("refreshImagesButton", label="Loading...", enabled=False)
        self.detail.reset()

        def inspect():
            index = listImages(path, self.shell.tasks.cancelEvent)
            return index, checkCaptions(index.folder, index.names, profile, self.shell.tasks.cancelEvent)

        def complete(result) -> None:
            if generation != self.loadGeneration or scope != self.scope or self.shell.isClosing:
                return
            index, checks = result
            layoutChanged = self.index is None or self.index.names != index.names
            self.index = index
            self.checks = checks if revision == self.editor.captionRevision else (checkCaptions(index.folder, index.names, self.profile) if self.profile.mode == "inline" else {})
            self.selected.intersection_update(index.names)
            if self.activeName not in index.names:
                self.activeName = index.names[0] if index.names else ""
            self.loading = False
            self.message = ""
            if self.visible:
                if layoutChanged:
                    self.render()
                else:
                    dearpygui.configure_item("refreshImagesButton", label="Refresh images", enabled=True)
                    self.drawRows()
                    self.detail.loadIfNeeded()
                    self.updateSelection()

        def failed(error) -> None:
            if generation != self.loadGeneration or scope != self.scope or self.shell.isClosing:
                return
            self.loading = False
            self.index = None
            self.message = str(error)
            self.render()

        self.loadFuture = self.shell.tasks.submit(inspect, complete, failed)

    def applyBulk(self, sender, appData, userData) -> None:
        if not self.visible or self.index is None:
            return
        self.detail.capture()
        names = tuple(self.index.names) if userData == "all" else tuple(sorted(self.selected))
        if not names or not self.activeName or self.profile.mode == "matching":
            return
        scope, profile, revision = self.scope, self.profile, self.editor.captionRevision
        value = profile.textCaptions.get(self.activeName, "") if profile.mode == "inline" else profile.fileAssignments.get(self.activeName, "")
        if profile.mode == "files" and not value.strip():
            raise ValidationError("Choose a caption file for the active image first.")
        actionName = "Use this caption" if profile.mode == "inline" and value.strip() else ("Clear captions" if profile.mode == "inline" else "Assign this file")

        def apply() -> None:
            if self.scope != scope or self.editor.captionRevision != revision or self.shell.view is not self.editor:
                raise ValidationError("The project or captions changed. Review the targets and apply again.")
            updated = profile.assignText(names, value) if profile.mode == "inline" else profile.assignFile(names, value)
            self.editor.setCaptions(updated)
            for name in names:
                state = CaptionCheck("Ready" if value.strip() else "Empty") if profile.mode == "inline" else CaptionCheck("Unchecked")
                self.updateCheck(name, state)
            self.shell.setStatus(f"Updated captions for {len(names)} image(s). External files were not changed.")
            self.refreshImages()

        targetLabel = "all" if userData == "all" else "selected"
        self.shell.confirm(actionName, f"{actionName} for {len(names)} {targetLabel} image(s)? Existing {'project text' if profile.mode == 'inline' else 'file assignments'} for these images will be replaced. Other caption modes and original files stay unchanged.", apply, confirmLabel="Apply changes")
