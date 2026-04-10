#!/usr/bin/env python3
"""Hardcoded batch runner that calls run_time_sweep.py for multiple scenes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# -------- Hardcoded configuration --------
# Replace these with your actual scene paths.
SCENES = [
    #"/mnt/DEV/DiplomaThesis/Scenes/bistro/bistro1-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/bistro/bistro2-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/emerald-square/emerald-square1-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/emerald-square/emerald-square2-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/modern-hall/modern-hall1-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/modern-hall/modern-hall2-power.pbrt",
    "/mnt/DEV/DiplomaThesis/Scenes/MPIIMODEL/MPII-view11-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/rungholt/rungholt1-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/rungholt/rungholt2-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/sponza-candles/sponza-candles1-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/sponza-candles/sponza-candles2-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/zero-day/measure-seven1-power.pbrt",
    #"/mnt/DEV/DiplomaThesis/Scenes/zero-day/measure-seven2-power.pbrt"
]

# This runtime set is applied to every scene listed above.
TIMES_SECONDS_FOR_ALL_SCENES = [1, 5, 10, 30, 60, 300, 600]

EXTRA_PBRT_ARGS: list[str] = ["--gpu"]
FAIL_FAST_WITHIN_SCENE = False
STOP_ON_ERROR = False

# ----------------------------------------

def scene_output_dir(base_dir: Path, scene_path: Path) -> Path:
    safe_name = scene_path.stem.replace(" ", "_")
    return base_dir / safe_name


def validate_times(times: list[float]) -> list[float]:
    cleaned: list[float] = []
    for seconds in times:
        if seconds <= 0:
            raise ValueError(f"All times must be > 0, got {seconds}")
        cleaned.append(seconds)
    if not cleaned:
        raise ValueError("At least one runtime must be configured")
    return cleaned


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    sweep_script = script_dir / "run_time_sweep.py"
    output_root_base = script_dir / "scene_output"

    if not sweep_script.is_file():
        print(f"Missing runner script: {sweep_script}", file=sys.stderr)
        return 1

    try:
        times = validate_times(TIMES_SECONDS_FOR_ALL_SCENES)
    except ValueError as exc:
        print(f"Invalid TIMES_SECONDS_FOR_ALL_SCENES: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for scene_str in SCENES:
        scene = Path(scene_str).expanduser().resolve()
        if not scene.is_file():
            print(f"[SKIP] Scene file not found: {scene}", file=sys.stderr)
            failures += 1
            if STOP_ON_ERROR:
                break
            continue

        cmd = [
            sys.executable,
            str(sweep_script),
            str(scene),
            *[f"{seconds:g}" for seconds in times],
            "--output-root",
            str(scene_output_dir(output_root_base, scene)),
        ]

        if FAIL_FAST_WITHIN_SCENE:
            cmd.append("--fail-fast")

        for arg in EXTRA_PBRT_ARGS:
            cmd.extend(["--pbrt-args", arg])

        print(f"\n=== Running scene: {scene} ===")
        print(f"Times (s): {', '.join(f'{seconds:g}' for seconds in times)}")
        print(" ".join(cmd))

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            if STOP_ON_ERROR:
                break

    print(f"\nFinished. Failed scene runs: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
