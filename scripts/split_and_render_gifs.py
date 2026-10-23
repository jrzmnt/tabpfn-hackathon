"""Splits an existing comparison_<env>.gif (Random | Expert | TabPFN-BC
panels, side by side, produced by record_comparison_video.py) into three
separate GIFs -- random_<env>.gif, expert_<env>.gif, tabpfn_bc_<env>.gif --
for placing side by side in the README. No re-rendering: every panel's
label is already baked into the pixels, so this is a pure crop + resave.

    uv run python scripts/split_and_render_gifs.py --env halfcheetah
    uv run python scripts/split_and_render_gifs.py --env hopper
"""
import argparse
from pathlib import Path

from PIL import Image, ImageSequence

from envs import ENVIRONMENTS

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"

PANEL_NAMES = ["random", "expert", "tabpfn_bc"]  # left-to-right order in comparison_<env>.gif


def split_existing_gif(path, n_panels):
    im = Image.open(path)
    frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    duration_ms = im.info.get("duration", 50)
    width, height = frames[0].size
    panel_width = width // n_panels
    panels = [
        [f.crop((i * panel_width, 0, (i + 1) * panel_width, height)) for f in frames]
        for i in range(n_panels)
    ]
    return panels, duration_ms


def save_gif(frames, path, duration_ms):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"saved {path} ({len(frames)} frames)")


def main(env_key):
    combined_gif = MEDIA_DIR / f"comparison_{env_key}.gif"
    panels, duration_ms = split_existing_gif(combined_gif, len(PANEL_NAMES))
    print(f"[{env_key}] split {combined_gif} into {len(PANEL_NAMES)} {len(panels[0])}-frame panels")

    for name, frames in zip(PANEL_NAMES, panels):
        save_gif(frames, MEDIA_DIR / f"{name}_{env_key}.gif", duration_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), default="halfcheetah",
                         help="which MuJoCo environment's comparison GIF to split")
    args = parser.parse_args()
    main(args.env)
