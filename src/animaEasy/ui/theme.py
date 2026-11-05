import dataclasses
import os
import pathlib

from dearpygui import dearpygui

from animaEasy.appearance import DEFAULT_ACCENT, blend, contrast, luminance, readableAccent


@dataclasses.dataclass(frozen=True)
class ThemeColors:
    background: tuple
    panel: tuple
    field: tuple
    border: tuple
    text: tuple
    muted: tuple
    sidebar: tuple
    button: tuple
    accent: tuple
    rawAccent: tuple
    selected: tuple
    primaryText: tuple
    warning: tuple
    error: tuple


def makeColors(mode: str, accent: tuple = DEFAULT_ACCENT) -> ThemeColors:
    if mode == "Light":
        background, panel, field, border = (245, 246, 249), (255, 255, 255), (255, 255, 255), (207, 210, 219)
        text, muted, sidebar, button = (33, 35, 43), (96, 101, 116), (234, 236, 242), (230, 233, 241)
        warning, error = (137, 87, 14), (169, 50, 62)
    else:
        background, panel, field, border = (25, 26, 30), (32, 33, 38), (27, 28, 33), (61, 63, 72)
        text, muted, sidebar, button = (233, 234, 239), (163, 166, 178), (21, 22, 26), (42, 44, 51)
        warning, error = (230, 185, 108), (241, 155, 151)
    foreground = (0, 0, 0) if contrast(accent, (0, 0, 0)) >= contrast(accent, (255, 255, 255)) else (255, 255, 255)
    selected = blend(panel, accent, 0.18)
    surface = (min if mode == "Light" else max)((background, panel, sidebar, selected), key=luminance)
    return ThemeColors(background, panel, field, border, text, muted, sidebar, button, readableAccent(accent, surface), accent, selected, foreground, warning, error)


colors = makeColors("Dark")


def definitions() -> list[tuple]:
    base = {dearpygui.mvThemeCol_WindowBg: colors.background, dearpygui.mvThemeCol_ChildBg: colors.panel, dearpygui.mvThemeCol_PopupBg: colors.panel, dearpygui.mvThemeCol_Text: colors.text, dearpygui.mvThemeCol_TextDisabled: colors.muted, dearpygui.mvThemeCol_Border: colors.border, dearpygui.mvThemeCol_FrameBg: colors.field, dearpygui.mvThemeCol_FrameBgHovered: blend(colors.field, colors.text, 0.07), dearpygui.mvThemeCol_FrameBgActive: blend(colors.field, colors.text, 0.11), dearpygui.mvThemeCol_Button: colors.button, dearpygui.mvThemeCol_ButtonHovered: blend(colors.button, colors.text, 0.08), dearpygui.mvThemeCol_ButtonActive: blend(colors.button, colors.rawAccent, 0.22), dearpygui.mvThemeCol_Header: colors.selected, dearpygui.mvThemeCol_HeaderHovered: blend(colors.panel, colors.rawAccent, 0.25), dearpygui.mvThemeCol_HeaderActive: blend(colors.panel, colors.rawAccent, 0.34), dearpygui.mvThemeCol_CheckMark: colors.accent, dearpygui.mvThemeCol_SliderGrab: colors.accent, dearpygui.mvThemeCol_SliderGrabActive: colors.rawAccent, dearpygui.mvThemeCol_Separator: colors.border, dearpygui.mvThemeCol_ScrollbarBg: colors.background, dearpygui.mvThemeCol_ScrollbarGrab: blend(colors.border, colors.text, 0.1), dearpygui.mvThemeCol_TextSelectedBg: (*colors.rawAccent, 80), dearpygui.mvThemeCol_TableHeaderBg: colors.field, dearpygui.mvThemeCol_TableRowBgAlt: (*colors.text, 5)}
    base.update({dearpygui.mvThemeCol_TitleBg: colors.panel, dearpygui.mvThemeCol_TitleBgActive: colors.selected, dearpygui.mvThemeCol_TitleBgCollapsed: colors.background, dearpygui.mvThemeCol_MenuBarBg: colors.panel, dearpygui.mvThemeCol_NavCursor: colors.accent, dearpygui.mvThemeCol_NavWindowingHighlight: (*colors.accent, 180), dearpygui.mvThemeCol_ResizeGrip: (*colors.rawAccent, 64), dearpygui.mvThemeCol_ResizeGripHovered: (*colors.rawAccent, 170), dearpygui.mvThemeCol_ResizeGripActive: colors.rawAccent, dearpygui.mvThemeCol_SeparatorHovered: colors.accent, dearpygui.mvThemeCol_SeparatorActive: colors.rawAccent, dearpygui.mvThemeCol_ScrollbarGrabHovered: blend(colors.border, colors.rawAccent, 0.3), dearpygui.mvThemeCol_ScrollbarGrabActive: colors.accent, dearpygui.mvThemeCol_Tab: colors.field, dearpygui.mvThemeCol_TabHovered: colors.selected, dearpygui.mvThemeCol_TabSelected: colors.selected, dearpygui.mvThemeCol_TabSelectedOverline: colors.accent, dearpygui.mvThemeCol_TabDimmed: colors.field, dearpygui.mvThemeCol_TabDimmedSelected: colors.panel, dearpygui.mvThemeCol_TabDimmedSelectedOverline: colors.accent, dearpygui.mvThemeCol_DockingPreview: (*colors.rawAccent, 180), dearpygui.mvThemeCol_DragDropTarget: colors.accent, dearpygui.mvThemeCol_DragDropTargetBg: (*colors.rawAccent, 48), dearpygui.mvThemeCol_InputTextCursor: colors.accent, dearpygui.mvThemeCol_PlotHistogram: colors.rawAccent, dearpygui.mvThemeCol_PlotHistogramHovered: colors.accent, dearpygui.mvThemeCol_PlotLines: colors.accent, dearpygui.mvThemeCol_PlotLinesHovered: colors.rawAccent})
    primary = {dearpygui.mvThemeCol_Button: colors.rawAccent, dearpygui.mvThemeCol_ButtonHovered: blend(colors.rawAccent, colors.text, 0.1), dearpygui.mvThemeCol_ButtonActive: blend(colors.rawAccent, colors.background, 0.12), dearpygui.mvThemeCol_Text: colors.primaryText}
    return [("appTheme", dearpygui.mvAll, True, base), ("primaryTheme", dearpygui.mvButton, True, primary), ("primaryTheme", dearpygui.mvButton, False, {dearpygui.mvThemeCol_Button: colors.button, dearpygui.mvThemeCol_Text: colors.muted}), ("selectedTheme", dearpygui.mvButton, True, {dearpygui.mvThemeCol_Button: colors.selected, dearpygui.mvThemeCol_Text: colors.accent}), ("sidebarTheme", dearpygui.mvChildWindow, True, {dearpygui.mvThemeCol_ChildBg: colors.sidebar}), ("pageTheme", dearpygui.mvChildWindow, True, {dearpygui.mvThemeCol_ChildBg: colors.background})]


def colorTag(themeTag: str, enabled: bool, item: int) -> str:
    return f"{themeTag}Color{int(enabled)}x{item}"


def installTheme(mode: str = "Dark", accent: tuple = DEFAULT_ACCENT) -> None:
    global colors
    colors = makeColors(mode, accent)
    for themeTag, component, enabled, values in definitions():
        if not dearpygui.does_item_exist(themeTag):
            dearpygui.add_theme(tag=themeTag)
        with dearpygui.theme_component(component, parent=themeTag, enabled_state=enabled):
            for item, color in values.items():
                dearpygui.add_theme_color(item, color, tag=colorTag(themeTag, enabled, item))
            if themeTag == "appTheme":
                dearpygui.add_theme_style(dearpygui.mvStyleVar_WindowPadding, 16, 16)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_FramePadding, 8, 5)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_ItemSpacing, 10, 8)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_CellPadding, 8, 7)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_FrameRounding, 4)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_FrameBorderSize, 1)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_ChildRounding, 5)
                dearpygui.add_theme_style(dearpygui.mvStyleVar_ScrollbarSize, 10)
    dearpygui.bind_theme("appTheme")


def updateTheme(mode: str, accent: tuple) -> None:
    global colors
    colors = makeColors(mode, accent)
    for themeTag, _component, enabled, values in definitions():
        for item, color in values.items():
            dearpygui.set_value(colorTag(themeTag, enabled, item), color)
    dearpygui.configure_viewport(0, clear_color=colors.background)


def installFonts() -> None:
    candidates = [pathlib.Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf", pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), pathlib.Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf")]
    fontPath = next((path for path in candidates if path.is_file()), None)
    if fontPath is None:
        return
    with dearpygui.font_registry():
        for tag, size in (("bodyFont", 16), ("titleFont", 26), ("captionFont", 13), ("brandFont", 22)):
            dearpygui.add_font(str(fontPath), size, tag=tag)
    dearpygui.bind_font("bodyFont")


def useFont(item: int | str, font: str) -> None:
    if dearpygui.does_item_exist(font):
        dearpygui.bind_item_font(item, font)
