import re
import shutil
import subprocess
import sys

from animaEasy.errors import ValidationError


DEFAULT_ACCENT = (111, 141, 245)
THEME_MODES = ("Auto", "Light", "Dark")
ACCENT_OPTIONS = {"Lapis": DEFAULT_ACCENT, "Azure": (65, 153, 232), "Violet": (160, 126, 235), "Rose": (222, 122, 161), "Teal": (62, 170, 162), "Amber": (218, 157, 60)}


def validateAppearance(mode: str, accent: tuple[int, int, int]) -> None:
    if mode not in THEME_MODES:
        raise ValidationError("Theme must be Auto, Light, or Dark.")
    if not isinstance(accent, (tuple, list)) or len(accent) != 3 or any(type(channel) is not int or not 0 <= channel <= 255 for channel in accent):
        raise ValidationError("Accent color must contain three color channels between 0 and 255.")


def systemTheme() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "Light" if value else "Dark"
        except OSError:
            return "Dark"
    if sys.platform.startswith("linux"):
        commands = [("gdbus", ["call", "--session", "--dest", "org.freedesktop.portal.Desktop", "--object-path", "/org/freedesktop/portal/desktop", "--method", "org.freedesktop.portal.Settings.Read", "org.freedesktop.appearance", "color-scheme"]), ("gsettings", ["get", "org.gnome.desktop.interface", "color-scheme"]), ("gsettings", ["get", "org.gnome.desktop.interface", "gtk-theme"])]
        for name, arguments in commands:
            executable = shutil.which(name)
            if executable is None:
                continue
            try:
                result = subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=0.6)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode:
                continue
            value = result.stdout.strip().lower()
            if name == "gdbus":
                match = re.search(r"uint32\s+([12])", value)
                if match:
                    return "Dark" if match[1] == "1" else "Light"
            elif "dark" in value:
                return "Dark"
            elif "light" in value:
                return "Light"
            elif arguments[-1] == "gtk-theme" and value.strip("\"' "):
                return "Light"
    return "Dark"


def resolveTheme(preference: str) -> str:
    if preference not in THEME_MODES:
        raise ValidationError("Choose Auto, Light, or Dark.")
    return systemTheme() if preference == "Auto" else preference


def blend(first: tuple, second: tuple, amount: float) -> tuple[int, int, int]:
    return tuple(round(left * (1 - amount) + right * amount) for left, right in zip(first[:3], second[:3], strict=True))


def luminance(color: tuple) -> float:
    channels = [value / 255 for value in color[:3]]
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
    return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def contrast(first: tuple, second: tuple) -> float:
    firstLight, secondLight = sorted((luminance(first), luminance(second)))
    return (secondLight + 0.05) / (firstLight + 0.05)


def readableAccent(accent: tuple, background: tuple) -> tuple[int, int, int]:
    destination = (255, 255, 255) if luminance(background) < 0.4 else (0, 0, 0)
    for step in range(21):
        candidate = blend(accent, destination, step / 20)
        if contrast(candidate, background) >= 4.5:
            return candidate
    return destination
