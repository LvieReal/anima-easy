import time

from dearpygui import dearpygui

from animaEasy.captions import MAX_CAPTION_LENGTH, CaptionCheck, CaptionDocument, captionSource, resolveCaption
from animaEasy.errors import ValidationError
from animaEasy.images import loadPreview
from animaEasy.ui import theme, widgets


class CaptionDetail:
    def __init__(self, owner):
        self.owner = owner
        self.shell = owner.shell
        self.preview = None
        self.previewKey = None
        self.previewIssue = ""
        self.document = None
        self.requestKey = None
        self.generation = 0
        self.future = None
        self.changedAt = 0.0

    def reset(self) -> None:
        self.preview = None
        self.previewKey = None
        self.previewIssue = ""
        self.resetCaption()

    def resetCaption(self) -> None:
        self.generation += 1
        if self.future is not None:
            self.future.cancel()
        self.requestKey = None
        self.document = None

    def key(self) -> tuple:
        owner = self.owner
        return (owner.scope, owner.activeName, owner.profile.mode, owner.profile.fileAssignments.get(owner.activeName, "") if owner.profile.mode == "files" else "")

    def tick(self) -> None:
        if time.monotonic() - self.changedAt >= 0.35:
            self.loadIfNeeded()

    def capture(self) -> None:
        if not self.owner.visible:
            return
        tag = "inlineCaptionInput" if self.owner.profile.mode == "inline" else "assignedCaptionPath"
        if dearpygui.does_item_exist(tag):
            callback = self.changeText if self.owner.profile.mode == "inline" else self.changeFile
            callback(tag, dearpygui.get_value(tag), dearpygui.get_item_configuration(tag)["user_data"])

    def draw(self) -> None:
        owner = self.owner
        if not owner.activeName:
            dearpygui.add_text("Choose an image to view its caption.", color=theme.colors.muted, wrap=0)
            return
        if self.requestKey != self.key():
            self.document = None
        compact = owner.galleryHeight() < 310
        with dearpygui.child_window(tag="captionPreviewArea", height=108, border=False, no_scrollbar=True, show=not compact):
            self.drawPreview()
        identity = (owner.scope, owner.activeName)
        if owner.profile.mode == "inline":
            widgets.labelWithHint(self.editorLabel(), f"Saved automatically inside this project, up to {MAX_CAPTION_LENGTH:,} characters per image. Original caption files are not edited.", "hintInlineCaption")
            dearpygui.add_input_text(tag="inlineCaptionInput", default_value=owner.profile.textCaptions.get(owner.activeName, ""), multiline=True, height=86 if compact else 90, width=-1, hint="Describe this image...", callback=self.changeText, user_data=identity)
        else:
            if owner.profile.mode == "files":
                widgets.pathField(self.shell, self.editorLabel(), "assignedCaptionPath", owner.profile.fileAssignments.get(owner.activeName, ""), False, self.changeFile, (".txt",), identity, str(owner.index.folder))
            else:
                source = captionSource(owner.index.folder, owner.activeName, owner.profile)
                dearpygui.add_text(widgets.readablePath(source, 65), color=theme.colors.muted, wrap=0)
            dearpygui.add_input_text(tag="captionReadOnly", default_value=self.document.text[:MAX_CAPTION_LENGTH] if self.document is not None else "", multiline=True, height=44 if compact else 80, width=-1, readonly=True, hint="Caption file preview (read-only)")
            with dearpygui.tooltip("captionReadOnly", delay=0.6, hide_on_activity=True):
                dearpygui.add_text("Read-only preview; original files stay unchanged. The first 16,384 characters are shown. Dataset checks read the full caption file, up to 1 MiB.", wrap=340)
        if owner.profile.mode != "matching":
            with dearpygui.group(horizontal=True):
                dearpygui.add_button(label=f"To selected ({len(owner.selected)})", tag="applyCaptionSelected", height=30, enabled=bool(owner.selected), callback=owner.applyBulk, user_data="selected")
                dearpygui.add_button(label=f"To all ({len(owner.index.names)})", tag="applyCaptionAll", height=30, callback=owner.applyBulk, user_data="all")
        self.loadIfNeeded()

    def editorLabel(self) -> str:
        label = "Caption" if self.owner.profile.mode == "inline" else "Caption file"
        return f"{label} · {widgets.readablePath(self.owner.activeName, 28)}"

    def resize(self) -> None:
        if not dearpygui.does_item_exist("captionPreviewArea"):
            return
        compact = self.owner.galleryHeight() < 310
        dearpygui.configure_item("captionPreviewArea", show=not compact)
        if dearpygui.does_item_exist("inlineCaptionInput"):
            dearpygui.configure_item("inlineCaptionInput", height=86 if compact else 90)
        if dearpygui.does_item_exist("captionReadOnly"):
            dearpygui.configure_item("captionReadOnly", height=44 if compact else 80)

    def render(self) -> None:
        if not self.owner.visible or not dearpygui.does_item_exist("captionDetail"):
            return
        dearpygui.delete_item("captionDetail", children_only=True)
        self.shell.releasePreviewTexture()
        with dearpygui.group(parent="captionDetail"):
            self.draw()

    def drawPreview(self) -> None:
        owner = self.owner
        currentPreview = self.preview if self.previewKey == (owner.scope, owner.activeName) else None
        currentIssue = self.previewIssue if self.previewKey == (owner.scope, owner.activeName) else ""
        with dearpygui.group(horizontal=True):
            if currentPreview is not None:
                dearpygui.add_static_texture(currentPreview.width, currentPreview.height, currentPreview.pixels, tag="captionPreviewTexture", parent="imageTextures")
                scale = min(1, 96 / currentPreview.height, 128 / currentPreview.width)
                dearpygui.add_image("captionPreviewTexture", width=round(currentPreview.width * scale), height=round(currentPreview.height * scale))
            else:
                with dearpygui.child_window(width=128, height=96, border=True, no_scrollbar=True):
                    dearpygui.add_text("No preview" if currentIssue else "Loading...", color=theme.colors.muted)
            with dearpygui.group():
                dearpygui.add_text(widgets.readablePath(owner.activeName, 30), wrap=210)
                if currentPreview is not None:
                    dearpygui.add_text(f"{currentPreview.originalSize[0]} × {currentPreview.originalSize[1]} px", color=theme.colors.muted)
                status = self.captionStatus()
                dearpygui.add_text(status, tag="activeCaptionStatus", color=theme.colors.muted if status == "Ready" else theme.colors.warning)
                issue = (self.document.issue if self.document is not None else "") or currentIssue
                if issue:
                    widgets.hint(issue, "hintCaptionIssue")

    def captionStatus(self) -> str:
        if self.owner.profile.mode == "inline":
            return "Ready" if self.owner.profile.textCaptions.get(self.owner.activeName, "").strip() else "Empty"
        return self.document.status if self.document is not None and self.requestKey == self.key() else "Reading..."

    def loadIfNeeded(self) -> None:
        owner = self.owner
        if not owner.visible or owner.index is None or not owner.activeName:
            return
        key = self.key()
        if key == self.requestKey:
            return
        self.generation += 1
        generation = self.generation
        self.requestKey = key
        if self.future is not None:
            self.future.cancel()
        name, folder, profile = owner.activeName, owner.index.folder, owner.profile
        cachedPreview = self.preview if self.previewKey == (owner.scope, name) else None
        self.document = None

        def read():
            preview, issue = cachedPreview, ""
            if preview is None:
                try:
                    preview = loadPreview(folder / name)
                except ValidationError as error:
                    issue = str(error)
            return preview, issue, resolveCaption(folder, name, profile)

        def complete(result) -> None:
            if generation != self.generation or key != self.key() or self.shell.isClosing:
                return
            self.preview, self.previewIssue, self.document = result
            self.previewKey = (owner.scope, name)
            if owner.profile.mode == "inline":
                text = owner.profile.textCaptions.get(name, "")
                self.document = CaptionDocument(text, "Project text", "Ready" if text.strip() else "Empty")
            owner.updateCheck(name, CaptionCheck(self.document.status, self.document.issue))
            if owner.visible and dearpygui.does_item_exist("captionPreviewArea"):
                dearpygui.delete_item("captionPreviewArea", children_only=True)
                self.shell.releasePreviewTexture()
                with dearpygui.group(parent="captionPreviewArea"):
                    self.drawPreview()
                if dearpygui.does_item_exist("captionReadOnly"):
                    dearpygui.set_value("captionReadOnly", self.document.text[:MAX_CAPTION_LENGTH])

        def failed(error) -> None:
            if generation == self.generation and key == self.key() and not self.shell.isClosing:
                self.previewIssue = str(error)
                self.document = CaptionDocument("", "", "Invalid", str(error))
                if owner.visible:
                    self.render()

        self.future = self.shell.tasks.submit(read, complete, failed)

    def restoreInput(self, sender, identity, value: str) -> None:
        if dearpygui.does_item_exist(sender) and dearpygui.get_item_configuration(sender).get("user_data") == identity:
            dearpygui.set_value(sender, value)

    def changeText(self, sender, appData, userData) -> None:
        scope, name = userData
        owner = self.owner
        if scope != owner.scope or owner.profile.mode != "inline" or not owner.visible:
            return
        if appData == owner.profile.textCaptions.get(name, ""):
            return
        try:
            profile = owner.profile.assignText((name,), appData)
        except ValidationError:
            self.restoreInput(sender, userData, owner.profile.textCaptions.get(name, ""))
            raise
        owner.editor.setCaptions(profile)
        state = CaptionCheck("Ready" if appData.strip() else "Empty")
        owner.updateCheck(name, state)
        if name == owner.activeName and dearpygui.does_item_exist("activeCaptionStatus"):
            dearpygui.set_value("activeCaptionStatus", state.status)
            dearpygui.configure_item("activeCaptionStatus", color=theme.colors.muted if state.status == "Ready" else theme.colors.warning)

    def changeFile(self, sender, appData, userData) -> None:
        scope, name = userData
        owner = self.owner
        if scope != owner.scope or owner.profile.mode != "files" or not owner.visible:
            return
        if appData == owner.profile.fileAssignments.get(name, ""):
            return
        try:
            profile = owner.profile.assignFile((name,), appData)
        except ValidationError:
            self.restoreInput(sender, userData, owner.profile.fileAssignments.get(name, ""))
            raise
        owner.editor.setCaptions(profile)
        owner.updateCheck(name, CaptionCheck("Unchecked"))
        if name == owner.activeName:
            self.changedAt = time.monotonic()
            self.resetCaption()
            dearpygui.set_value("captionReadOnly", "")
            dearpygui.set_value("activeCaptionStatus", "Reading...")
