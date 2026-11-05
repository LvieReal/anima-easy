from dearpygui import dearpygui

from animaEasy.domain import EFFORT_PRESETS
from animaEasy.ui import theme, widgets


BASIC_FIELDS = (("rank", "LoRA rank", 1, 512), ("alpha", "LoRA alpha", 1, 512), ("learningRate", "Learning rate", 0.00000001, 1.0), ("epochs", "Epochs", 1, 10000), ("resolution", "Resolution", 256, 2048), ("repeats", "Dataset repeats", 1, 10000))
ADVANCED_FIELDS = (("batchSize", "Batch size", 1, 64), ("gradientAccumulation", "Gradient accumulation", 1, 256), ("seed", "Seed", 0, 2147483647), ("mixedPrecision", "Mixed precision", 0, 0), ("blocksToSwap", "Blocks to swap", 0, 26), ("optimizer", "Optimizer", 0, 0), ("vaeChunkSize", "VAE chunk size", 0, 512), ("keepCheckpoints", "Epoch checkpoints to keep", 1, 50))
TRAINING_HINTS = {
    "effortPreset": "Presets change only epochs and dataset repeats. Low is 5 epochs with 5 repeats; Medium is 10 with 10; High is 20 with 10. Resolution, rank, and memory options stay unchanged. More training is not always better: too much can overfit. Custom keeps your current values.",
    "rank": "How much new information the LoRA can learn. A larger rank can capture more detail, but increases file size and memory use. Start modestly and increase it if important features are consistently missing.",
    "alpha": "Scales the LoRA's training updates relative to its rank. Keeping alpha equal to rank is a straightforward starting point. If you change it, compare results carefully because the effective strength of the updates changes too.",
    "learningRate": "How much the model changes at each update. Too high can make training unstable or wash out useful details; too low may learn very slowly. Make small adjustments rather than large jumps.",
    "epochs": "How many times training goes through the repeated dataset. More epochs give the LoRA more opportunities to learn, but too many can make it memorize the examples. Compare saved checkpoints instead of assuming the last is best.",
    "resolution": "The target training image resolution. Larger values can preserve finer details but require more memory and time. Aspect-ratio buckets can preserve different image shapes rather than forcing every image into a square.",
    "repeats": "How often each image is counted inside one epoch. For example, 20 images with 5 repeats give 100 image uses per epoch. Small datasets often use more repeats; large datasets usually need fewer.",
    "batchSize": "How many images are processed together. Bigger batches need more GPU memory. A batch size of 1 is a conservative starting point, especially when memory is limited.",
    "gradientAccumulation": "Combines several small batches before updating the model. The effective batch size is batch size times accumulation. This can act like a larger batch without holding all those images in GPU memory at once.",
    "seed": "Controls random choices such as shuffling. Reusing a seed makes similar runs easier to compare, although GPU operations and software changes can still affect the result. Change it when you want a different random sequence.",
    "mixedPrecision": "The number format used for many training computations. BF16 is a common choice on supported modern NVIDIA GPUs. FP16 is another memory-saving option but is more sensitive to numerical overflow. No mixed precision generally needs more memory.",
    "gradientCheckpointing": "Saves GPU memory by recalculating some intermediate results instead of storing them. It usually makes training a little slower, but can let a model fit when memory is tight.",
    "blocksToSwap": "Moves transformer blocks between system RAM and the GPU to reduce GPU memory use. More swapped blocks need more system RAM and usually train more slowly. Zero disables swapping. Standard Anima models allow up to 26; custom models may allow fewer. Choose the amount from actual memory use, not GPU capacity alone.",
    "optimizer": "AdamW is the standard optimizer. AdamW8bit stores optimizer statistics more compactly using bitsandbytes, which can save GPU memory. Both save optimizer state for resume. Changing the optimizer requires a new run.",
    "vaeChunkSize": "Processes the VAE in spatial chunks to reduce peak encoding memory. Zero disables chunking; positive values must be even. The default is 128. Smaller chunks can trade speed for memory. This affects image encoding, not the main transformer block count.",
    "keepCheckpoints": "Saves LoRA weights and full trainer state after each completed epoch, retaining this many epoch checkpoints in the current run. Older checkpoints from that run are removed. Final weights and final state are saved separately. Original model files and other runs are not deleted.",
    "cacheLatents": "Encodes training images once and reuses their compact representations. This avoids repeating image-encoding work. Cached results are written only inside the new run folder, never next to your original images. Each run gets its own cache.",
    "cacheTextEncoder": "Reuses encoded representations of your captions instead of encoding the same text repeatedly. This can reduce work and memory use. The run captures the current caption text and writes its own disk cache. Later project edits do not change that run or its resume inputs.",
}


def presetDescription(settings) -> str:
    if settings.effortPreset == "Custom":
        return "Custom effort. Adjust epochs and repeats below; other settings are independent."
    return f"{settings.effortPreset}: {settings.epochs} epochs × {settings.repeats} repeats = {settings.epochs * settings.repeats} image uses per source image."


def drawFields(editor, fields: tuple) -> None:
    with dearpygui.table(header_row=False, policy=dearpygui.mvTable_SizingStretchSame):
        dearpygui.add_table_column()
        dearpygui.add_table_column()
        for offset in range(0, len(fields), 2):
            with dearpygui.table_row():
                for name, label, minimum, maximum in fields[offset:offset + 2]:
                    with dearpygui.group():
                        widgets.labelWithHint(label, TRAINING_HINTS[name], f"hint{name[0].upper()}{name[1:]}")
                        value = getattr(editor.project.training, name)
                        options = {"tag": f"training{name[0].upper()}{name[1:]}", "width": -1, "default_value": value, "callback": editor.changeTraining, "user_data": name}
                        if name == "resolution":
                            dearpygui.add_combo([str(number) for number in range(256, 2049, 64)], **{**options, "default_value": str(value)})
                        elif name == "mixedPrecision":
                            dearpygui.add_combo(["bf16", "fp16", "no"], **options)
                        elif name == "optimizer":
                            dearpygui.add_combo(["AdamW", "AdamW8bit"], **options)
                        elif name == "learningRate":
                            dearpygui.add_input_double(min_value=minimum, max_value=maximum, min_clamped=True, max_clamped=True, step=0, format="%.8f", **options)
                        else:
                            dearpygui.add_input_int(min_value=minimum, max_value=maximum, min_clamped=True, max_clamped=True, step=2 if name == "vaeChunkSize" else 1, **options)


def drawTrainingForm(editor) -> None:
    widgets.labelWithHint("Training effort", TRAINING_HINTS["effortPreset"], "hintEffortPreset")
    dearpygui.add_combo(list(EFFORT_PRESETS), default_value=editor.project.training.effortPreset, width=230, tag="trainingPreset", callback=editor.changePreset)
    dearpygui.add_text(presetDescription(editor.project.training), tag="presetDescription", color=theme.colors.muted, wrap=0)
    dearpygui.add_spacer(height=2)
    drawFields(editor, BASIC_FIELDS)
    dearpygui.add_spacer(height=4)
    with dearpygui.collapsing_header(label="Memory and reproducibility", tag="advancedTraining", default_open=False):
        drawFields(editor, ADVANCED_FIELDS)
        for name, label in (("gradientCheckpointing", "Gradient checkpointing"), ("cacheLatents", "Cache latents"), ("cacheTextEncoder", "Cache text encoder outputs")):
            with dearpygui.group(horizontal=True):
                dearpygui.add_checkbox(label=label, tag=f"training{name[0].upper()}{name[1:]}", default_value=getattr(editor.project.training, name), callback=editor.changeTraining, user_data=name)
                widgets.hint(TRAINING_HINTS[name], f"hint{name[0].upper()}{name[1:]}")
    dearpygui.add_spacer(height=6)
    dearpygui.add_text("Hover over (?) for guidance. Higher effort does not guarantee better results.", color=theme.colors.muted, wrap=0)
