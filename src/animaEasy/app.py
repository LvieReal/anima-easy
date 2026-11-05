import argparse
import logging
import logging.handlers
import os
import pathlib
import sys

import filelock
import platformdirs

from animaEasy import VERSION
from animaEasy.errors import AnimaError
from animaEasy.workspace import Workspace


def main() -> int:
    parser = argparse.ArgumentParser(description="Anima Easy: a guided, local desktop workspace for Anima LoRA projects.")
    parser.add_argument("--data-dir", dest="dataDir", type=pathlib.Path, help="Use a separate application-state directory (useful for portable installs and testing).")
    parser.add_argument("--version", action="version", version=f"Anima Easy {VERSION}")
    arguments = parser.parse_args()
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        print("Anima Easy needs a graphical session. Start it from an X11 desktop or a Wayland session with XWayland enabled.", file=sys.stderr)
        return 1
    from animaEasy.ui.shell import Desktop, showStartupError

    statePath = (arguments.dataDir or platformdirs.user_data_path("AnimaEasy", appauthor=False)).expanduser().resolve()
    desktop = None
    try:
        statePath.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(statePath / "session.log", maxBytes=1024 * 1024, backupCount=1, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(message)s")
        with filelock.FileLock(statePath / "session.lock", timeout=0):
            workspace = Workspace(statePath)
            desktop = Desktop(workspace)
            try:
                desktop.run()
            finally:
                desktop.close()
    except filelock.Timeout:
        showStartupError("Another Anima Easy window is already using this application-state directory. Close it before opening another window with the same --data-dir.")
        return 1
    except (AnimaError, OSError) as error:
        logging.exception("Anima Easy could not complete the session")
        print(str(error), file=sys.stderr)
        if desktop is None:
            showStartupError(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
