import pathlib

from dearpygui import dearpygui

from animaEasy.errors import ValidationError
from animaEasy.ui import theme


def pageHeading(eyebrow: str, title: str, subtitle: str) -> None:
    theme.useFont(dearpygui.add_text(eyebrow, tag="pageKicker", color=theme.colors.accent), "captionFont")
    theme.useFont(dearpygui.add_text(title), "titleFont")
    dearpygui.add_text(subtitle, color=theme.colors.muted, wrap=0)
    dearpygui.add_spacer(height=4)
    dearpygui.add_separator()
    dearpygui.add_spacer(height=4)


def section(title: str, subtitle: str = "") -> None:
    dearpygui.add_text(title)
    if subtitle:
        dearpygui.add_text(subtitle, color=theme.colors.muted, wrap=0)
    dearpygui.add_spacer(height=2)


def hint(text: str, tag: str) -> None:
    dearpygui.add_text("(?)", tag=tag, color=theme.colors.muted)
    with dearpygui.tooltip(tag, tag=f"{tag}Tooltip", delay=0.25, hide_on_activity=True):
        dearpygui.add_text(text, wrap=360)


def labelWithHint(label: str, explanation: str, tag: str) -> None:
    with dearpygui.group(horizontal=True):
        dearpygui.add_text(label)
        hint(explanation, tag)


def primaryButton(label: str, **options) -> int | str:
    item = dearpygui.add_button(label=label, height=36, **options)
    dearpygui.bind_item_theme(item, "primaryTheme")
    return item


def pathField(shell, label: str, tag: str, value: str, directory: bool, onChange=None, fileExtensions: tuple[str, ...] = (".safetensors",), userData=None, browseFolder: str = "") -> None:
    dearpygui.add_text(label)
    with dearpygui.group(horizontal=True):
        dearpygui.add_input_text(tag=tag, default_value=value, width=-104, hint="Choose a folder" if directory else f"Choose a {fileExtensions[0]} file", callback=onChange, user_data=userData)
        dearpygui.add_button(label="Browse...", tag=f"{tag}Browse", width=94, callback=lambda: shell.browse(tag, directory, onChange, fileExtensions, userData, browseFolder))


def details(rows: list[tuple[str, str]]) -> None:
    with dearpygui.table(header_row=False, resizable=False, policy=dearpygui.mvTable_SizingStretchProp, borders_innerH=True):
        dearpygui.add_table_column(init_width_or_weight=180, width_fixed=True)
        dearpygui.add_table_column()
        for label, value in rows:
            with dearpygui.table_row():
                dearpygui.add_text(label, color=theme.colors.muted)
                dearpygui.add_text(value, wrap=0)


def formatSize(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.2f} GiB"
    if size >= 1024 ** 2:
        return f"{size / 1024 ** 2:.1f} MiB"
    return f"{size / 1024:.1f} KiB"


def readablePath(path: str, limit: int = 90) -> str:
    if len(path) <= limit:
        return path
    return f"...{path[-limit + 3:]}"


def defaultBrowsePath(value: str, basePath: str = "") -> str:
    if not value.strip():
        value = basePath or str(pathlib.Path.home())
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute() and basePath:
        path = pathlib.Path(basePath) / path
    if path.is_file():
        path = path.parent
    return str(path) if path.is_dir() else str(pathlib.Path.home())


def selectedDialogPath(data: dict, directory: bool) -> str:
    selections = data.get("selections", {})
    if not directory and selections:
        if not isinstance(selections, dict) or len(selections) != 1:
            raise ValidationError("Choose one existing file.")
        value = next(iter(selections.values()))
    else:
        value = data.get("file_path_name", "")
    if not value:
        return ""
    path = pathlib.Path(value)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValidationError("The selected folder no longer exists." if directory else "Choose an existing file. Its original extension must be preserved.")
    return str(path)
