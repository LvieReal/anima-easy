# Anima Easy

A native Anima LoRA trainer and project workspace, built with Dear PyGui. Keep separate projects, reference existing model files, organize captions, and run one local training job without a browser or web server. Training dependencies are installed separately from the desktop UI.

## Start the app

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13.7 or newer**, then extract or clone this repository. Check an existing installation with `uv --version`.

- **Windows:** double-click **`runWindows.bat`**.
- **Linux:** run **`./runLinux.sh`**. In a file manager, enable “executable” or “Run as a program” if needed. If permissions were lost while extracting an archive, run `chmod +x runLinux.sh` once.

Both launchers find their own project directory, so they also work through shortcuts or from another working directory. The first launch downloads Python 3.14 and the locked packages if needed; later launches reuse the local environment. Managed training setup is a separate, explicit download step. Model weights are never downloaded automatically.

The equivalent terminal command, run from the repository, is:

```sh
uv run --locked --no-dev anima-easy
```

This is a source distribution, not a standalone installer. Keep the launcher beside `pyproject.toml`, `uv.lock`, and `src`.

### Desktop requirements

- **Windows:** a desktop session and a working DirectX 11 graphics driver.
- **Linux:** X11 or XWayland and an OpenGL 3-capable driver. Minimal Debian/Ubuntu installations may need `libgl1`, `libx11-6`, `libxrandr2`, `libxinerama1`, `libxcursor1`, and `libxi6`.

The launcher forwards command-line arguments. For example, to use separate application state:

```sh
./runLinux.sh --data-dir "/absolute/path/to/application state"
```

```bat
runWindows.bat --data-dir "D:\Anima Easy state"
```

Relative argument paths are resolved from the repository directory. Windows keeps the console open after an error so you can read the message. If a Linux file manager does not execute shell scripts, run the launcher from a terminal.

## Set up a project

1. Choose a workspace folder. Changing it selects another project collection; it never moves existing projects.
2. Register existing `.safetensors` files as diffusion model, text encoder, and VAE. Choose an Anima family for diffusion models. Files stay in place, including files shared with another application. You can skip registration during setup.
3. Create a project and follow **Dataset → Models → Training settings → Review**.
4. Select a flat dataset folder. Supported images are JPG, JPEG, PNG, WebP, BMP, and GIF. Subfolders are ignored.
5. Choose a caption source, then select the model references and training settings for this project.

Valid edits save automatically after a short delay, on navigation, and on normal close. **Ctrl+S** or **Save now** saves immediately. Use different project names for clarity; folder identity uses a UUID rather than the name.

Before starting a run, set up the managed runtime from **Training** in the sidebar and pass its GPU check. Projects can still be edited without installing the training runtime or having a CUDA GPU.

## Manage captions

Choose one of three **Caption source** modes on the Dataset page:

| Source | Behavior |
| --- | --- |
| **Matching .txt files** | `image.png` uses `image.txt` in the same folder. The caption preview is read-only. |
| **Assigned caption files** | Browse for a caption file for each image. Its name and location do not have to match the image. One file can be assigned to several images. |
| **Text inside the project** | Type a separate caption for each image. Text is saved inside `project.json`; no sidecar files are written. |

Click an image name to open its thumbnail and caption. Use the checkboxes to choose bulk targets; opening an image does not select it for bulk changes.

For a shared caption, edit the active image's text or assign its caption file, then use **To selected** or **To all**. Confirm the target count before existing text or assignments are replaced. Applying an empty project caption clears the targeted captions. **To all includes images hidden by filters or pagination.** “Select page” selects only the visible page; selections remain checked across pages and filters.

Switching sources preserves each mode's text and file assignments. Caption profiles are separate for each dataset folder path, so switching to another folder does not silently attach the previous folder's captions to same-named images. Returning to the original folder restores its profile. Renamed or moved folders count as different paths; keep the original folder path if you want to reuse that profile.

- Inline captions allow up to 16,384 characters per image.
- Relative paths typed into a caption-file field are resolved against the dataset folder. The file browser records absolute paths.
- External captions must be UTF-8, with an optional BOM. Checks accept files up to 1 MiB; previews show the first 16,384 characters.
- Images and external caption files are never rewritten, renamed, copied, or deleted by the caption editor.
- Use **Refresh images** after adding, renaming, deleting, or externally editing files. The list is paginated to keep the interface responsive; only the active image's thumbnail is loaded.
- **Check dataset** reads the selected caption source and checks image files and missing, empty, or invalid captions. Different image extensions may share a stem and the same matching caption file. Reports are snapshots; caption changes invalidate them, and external file changes require a recheck.

## Training settings

Every setting has a **(?)** hover explanation, including the options under **Memory and reproducibility**.

The **Training effort** presets change only epochs and dataset repeats:

| Preset | Epochs | Repeats | Image uses per source image |
| --- | ---: | ---: | ---: |
| Custom | Your values | Your values | Depends on your values |
| Low | 5 | 5 | 25 |
| Medium | 10 | 10 | 100 |
| High | 20 | 10 | 200 |

These are relative effort levels, not memory profiles or quality promises. Rank, resolution, learning rate, batch size, and memory options remain unchanged. Editing epochs or repeats switches the preset to Custom. Choosing Custom keeps the current values. Higher effort can overfit; appropriate settings depend on the images, captions, and training goal.

## Set up and run training

1. Open **Training → Managed runtime**. Choose a **Runtime storage folder** before setup, such as `D:\Anima training` or `E:\Anima training` on Windows. The app manages an `anima-v1` subfolder there. **Use folder** saves the choice; **Set up runtime** also applies the path currently in the field before asking for confirmation. Check the destination in that confirmation. Setup installs Python **3.12.16**, PyTorch **2.9.1 with CUDA 12.8**, and locked dependencies in that managed subfolder. It downloads the Apache-2.0 [sd-scripts trainer](https://github.com/kohya-ss/sd-scripts/tree/cb70b96858477e716a2096292ab8aab433277f38) at revision `cb70b96858477e716a2096292ab8aab433277f38` and verifies its archive hash. It does not modify the UI environment, install NVIDIA drivers, or copy model weights.
2. Allow several GB of downloads from GitHub, PyPI, and the official PyTorch package index. Keep at least **12 GiB free for setup**, or **24 GiB on drives without hardlinks** such as exFAT, plus space for projects and checkpoints. The runtime's managed Python, uv download cache, Hugging Face/PyTorch caches, and temporary files use the selected runtime storage. Project run caches and checkpoints use the project workspace, so choose a workspace on a suitable drive too. Internet access is needed for uncached setup packages; the trainer uses bundled tokenizers rather than downloading models.
3. Use **Check GPU** after changing your driver or hardware. Training requires an **x86-64 Windows or Linux system and a usable NVIDIA CUDA GPU**. The check imports the actual trainer and runs a small CUDA kernel when a GPU is available. It is not a training-memory or model-compatibility test.
4. Register compatible Anima diffusion, Qwen3 text-encoder, and VAE `.safetensors` files. Select them in the project's Models step. A successful file registration checks structure, not whether those three model components work together.
5. Review the dataset and settings, then choose **Start training** on Review. The app checks CUDA again, hashes referenced images and models, captures the current captions and settings, and launches the pinned trainer. Missing or empty captions use empty text; invalid caption files block the run.

If you do not change the location, the runtime defaults to `<application state>/training/anima-v1`—normally `%LOCALAPPDATA%\AnimaEasy\training\anima-v1` on Windows. Selecting another folder does not move or delete an existing installation. Choose the parent storage folder to reuse an existing managed runtime, or set up a new one in the new location. Runtime location changes are blocked during managed operations. Application preferences, model references, and resume-ownership keys stay in the application-state folder; the UI environment stays beside the launcher.

The Training page shows actual trainer step counts and loss, live output, and saved runs. Loading models and preparing caches have no training-step percentage. **Follow output** keeps the log at its newest lines; turn it off to scroll. **Outputs / full log** opens the selected run's folder, and **Captured inputs** opens its saved configuration. There is one managed operation at a time and no queue. You can edit projects while training, but workspace switching is blocked until the operation stops.

### Python and installer compatibility

The desktop uses Python 3.14; the training runtime uses a separate Python 3.12.16 environment with its tested, locked dependency set. You do not need to install it globally.

uv bundles its Python download catalog. An older uv can report **No interpreter found for Python 3.12.16 in managed installations** even though that Python build is available. Managed setup checks uv before downloading or replacing runtime files. Update a standalone uv installation with `uv self update`; if you installed uv through a package manager, update it there instead. Confirm `uv --version` is at least 0.13.7, close and reopen Anima Easy, and retry **Set up runtime** or **Repair setup**. The chosen runtime storage folder is retained.

On Windows exFAT/FAT32 storage, if uv downloaded Python but could not create its optional minor-version junction, retry **Set up runtime** or **Repair setup**. Setup can reuse the exact downloaded interpreter directly in the selected folder, without moving Python to another drive. Drives without hardlinks use package copies and need more free space. FAT32's 4 GiB per-file limit still applies; use NTFS or exFAT for larger files. NTFS supports links and avoids the extra package copies.

### Choose memory settings from actual runs

GPU capacity alone does not determine the right resolution. An 8 GB GPU does not imply a 512 px limit: keep 896 or 1024 px if that resolution works with your models and settings. There is no universal fit guarantee. Batch size 1, disk caches, gradient checkpointing, and block swapping can reduce memory use; block swapping uses system RAM and trades speed for lower GPU-memory use. A smaller even VAE chunk size can reduce cache-encoding memory use. AdamW8bit is an optional optimizer; AdamW is the default. Close other GPU-heavy applications before testing a run.

Memory use depends on the exact models, image buckets, precision, LoRA rank, driver, and settings. The effort presets do not adjust memory settings. If a run exhausts VRAM, inspect its full log and start a new run with more conservative settings; resuming preserves the original configuration.

### Stop and resume

- **Stop operation** asks for confirmation. Keep the app open during training; closing it stops its managed process.
- LoRA weights and full trainer state are saved after completed epochs. The selected retention count keeps that many epoch checkpoints in the current run, plus final weights and state. Older checkpoints from other runs are not removed.
- Work since the last fully saved epoch is lost on stop or failure. A run stopped before its first complete epoch state has nothing to resume from.
- In **Saved runs and checkpoints**, choose a stopped run and a saved epoch, then **Resume saved epoch**. This restores model, optimizer, scheduler, and random-generator state. It creates a new run folder; the earlier run is preserved.
- Resume uses the original captured captions, settings, and epoch target, not later project edits. The referenced image and model bytes must still match their captured hashes. A state already at its target is not offered for further training. Start a new run to change the target or configuration.
- Resume accepts only intact states authenticated by this application profile. Keep the matching ownership key and app/runtime version; foreign, modified, or incompatible states are rejected. A saved state is not a promise of bitwise-identical results across GPUs or software environments.

**Repair setup** reinstalls the pinned runtime and checks it again. Use it after an interrupted installation or if runtime files were changed. Do not manually upgrade packages inside the managed environment. Repair does not delete project runs or reinstall NVIDIA drivers.

## Appearance

Open **Settings** in the sidebar:

1. **Theme:** Auto, Light, or Dark. Dark gray is the default. Auto follows the operating system's app preference when available and falls back to Dark. System changes are checked periodically in the background.
2. **Accent color:** choose a named color or use the custom color picker. Lapis is the default. Accent text is adjusted for readability without changing the stored color.

Appearance preferences apply across projects and persist when switching workspaces.

## Storage and recovery

Application state uses the per-user `AnimaEasy` directory selected by platformdirs: `%LOCALAPPDATA%\AnimaEasy` on Windows; `$XDG_DATA_HOME/AnimaEasy` or `~/.local/share/AnimaEasy` on Linux. It contains `settings.json`, `models.json`, a bounded `session.log`, a session lock, and the managed `training/` directory. Run only one app instance per state directory.

Projects live at `<workspace>/projects/<projectId>/project.json`. Each stores its settings, shared model IDs, and per-folder caption profiles. Copying a project record does not copy referenced datasets, caption files, or model weights. Back up both the project files and any external files you need.

Each run lives at `<workspace>/projects/<projectId>/runs/<runId>/`. It contains `run.json`, `training.log`, captured `inputs/`, run-owned `cache/`, and `outputs/`. Captions and settings are copied into the run, but images and base-model weights remain referenced in their original locations. Do not edit those files during a run or remove files needed for resume. The final LoRA is `outputs/anima.safetensors`; `*-state` folders contain trainer state, not just LoRA weights.

Back up the workspace, external images/models, and application state. **For resume, preserve `training/state.key` and `training/stateIdentity.json` together.** Keep the key private. Copying only a project or workspace to a new application profile is not enough to authenticate its saved states. The large `training/anima-v1` runtime can be reinstalled using the matching app version; the ownership key cannot be recreated to unlock old states. Never delete or replace a damaged key to try to repair existing runs.

After a run stops, its `cache/` folder may be removed to reclaim space; a resumed run prepares its own cache. Keep the captured inputs and chosen trainer-state folder intact. Run folders and checkpoint metadata can contain captions and local paths; review what you share.

Model registration checks the safetensors structure without loading weight tensors. It does not verify architecture, numerical integrity, licensing, or training compatibility. Changed or missing files are flagged; register a changed file again to recheck it. Forgetting a model removes only the registry entry, never the original file.

Writes use temporary files and atomic replacement. Invalid or unsupported state is reported rather than reset. Project saves detect stale edits and serialize concurrent writers with a file lock. If a project cannot save during normal shutdown, the app attempts to preserve its in-memory draft under `<application state>/recovery/` and reports the location in the log and terminal. It does not overwrite a newer project to recover that draft.

## Development commands

```sh
uv sync --locked --group dev
uv run ruff check .
uv run pytest -m "not desktop and not runtime"
```

With a desktop session, `uv run pytest` also runs native viewport tests. On a headless Linux host with Xvfb, xauth, xdotool, and Mesa installed:

```sh
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run pytest
```

To run the optional real-backend CPU contracts, set `ANIMA_EASY_TEST_RUNTIME` to an installed `<application state>/training/anima-v1` directory and run `uv run pytest -m runtime`. Include a display and xdotool for its native managed-operation check. These contracts do not run full Anima training or measure GPU-memory fit.

The UI renders at up to 30 FPS while active and 8 FPS while idle. Filesystem inspection runs on a worker, and UI updates stay on the main thread. System Segoe UI, DejaVu Sans, or Liberation Sans is used when available, with Dear PyGui's default font as fallback.

Application identifiers use camelCase and UPPER_SNAKE_CASE. Python special methods, third-party APIs, and configuration keys required by tools retain their original names.

## License and attribution

MIT. Inspired by [Citron's Anima LoRA Trainer](https://github.com/citronlegacy/citron-anima-lora-trainer-ui). See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for attribution and third-party licensing.
