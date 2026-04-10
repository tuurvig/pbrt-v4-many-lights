#!/usr/bin/env python3
"""Run pbrt multiple times using explicit render-time targets."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSpec:
    seconds: float
    run_dir: Path


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Allow --pbrt-args values that start with '-' (e.g. '--gpu')."""
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--pbrt-args" and i + 1 < len(argv):
            normalized.append(f"--pbrt-args={argv[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def parse_time_values(raw_values: list[str]) -> list[float]:
    values: list[float] = []
    for raw in raw_values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed = float(token)
            except ValueError as exc:
                raise ValueError(f"Invalid time value: '{token}'") from exc
            if parsed <= 0:
                raise ValueError(f"Time values must be > 0, got {parsed}")
            values.append(parsed)
    if not values:
        raise ValueError("At least one time value is required")
    return values


def seconds_label(seconds: float) -> str:
    if seconds.is_integer():
        return str(int(seconds))
    text = f"{seconds:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pbrt once per target time value, passing each directly as "
            "--render-time={seconds}."
        )
    )
    parser.add_argument("scene", type=Path, help="Path to scene .pbrt file")
    parser.add_argument(
        "times",
        nargs="+",
        help="Target times in seconds (examples: 5 10 30 or 5,10,30)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory that will contain one subfolder per run",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first non-zero return code",
    )
    parser.add_argument(
        "--pbrt-args",
        default="",
        help=(
            "Additional arguments passed through to pbrt as a quoted string "
            '(example: --pbrt-args "--gpu --seed 1")'
        ),
    )
    parser.add_argument(
        "--pbrt-arg",
        action="append",
        default=[],
        help="Single pbrt argument. Repeat this flag for multiple arguments.",
    )
    return parser


def build_run_specs(times: list[float], output_root: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seconds in times:
        run_name = f"{seconds_label(seconds)}s"
        specs.append(RunSpec(seconds=seconds, run_dir=output_root / run_name))
    return specs


def pick_primary_exr(run_dir: Path) -> Path | None:
    exr_files = [p for p in run_dir.glob("*.exr") if p.is_file()]
    if not exr_files:
        return None
    return max(exr_files, key=lambda p: p.stat().st_mtime)


def rename_outputs_to_exr_name(run_dir: Path, exr_file: Path) -> None:
    base = exr_file.stem
    for name in ("console.log", "command.sh"):
        src = run_dir / name
        if not src.is_file():
            continue
        target = run_dir / f"{base}{src.suffix}"
        if target == src:
            continue

        if target.exists() and target != src:
            target.unlink()
        src.rename(target)


def main() -> int:
    parser = make_parser()
    args = parser.parse_args(normalize_cli_args(sys.argv[1:]))

    try:
        times = parse_time_values(args.times)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene file not found: {scene}")

    script_dir = Path(__file__).resolve().parent
    binary = script_dir / "pbrt"
    if not binary.is_file():
        parser.error(f"pbrt binary not found next to script: {binary}")
    if not os.access(binary, os.X_OK):
        parser.error(f"pbrt binary is not executable: {binary}")

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else (script_dir / Path("scene_output"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    pbrt_args: list[str] = []
    if args.pbrt_args:
        pbrt_args.extend(shlex.split(args.pbrt_args))
    if args.pbrt_arg:
        pbrt_args.extend(args.pbrt_arg)

    run_specs = build_run_specs(times, output_root)
    failures = 0

    print(f"Scene: {scene}")
    print(f"Binary: {binary}")
    print(f"Output root: {output_root}")
    print("")

    for spec in run_specs:
        run_dir = spec.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        command_parts = [
            shlex.quote(str(binary)),
            "--stats",
            f"--render-time={spec.seconds:g}",
            *[shlex.quote(arg) for arg in pbrt_args],
            shlex.quote(str(scene)),
        ]
        pbrt_cmd = " ".join(command_parts)
        shell_cmd = f"{pbrt_cmd} > console.log 2>&1"

        (run_dir / "command.sh").write_text(shell_cmd + "\n", encoding="utf-8")

        print(f"[{spec.seconds:g}s -> render-time={spec.seconds:g}] {run_dir}")

        result = subprocess.run(
            ["/bin/bash", "-lc", shell_cmd],
            cwd=run_dir,
            check=False,
        )

        exr_file = pick_primary_exr(run_dir)
        if exr_file is not None:
            rename_outputs_to_exr_name(run_dir, exr_file)
        else:
            print("  warning: no .exr file found; skipping post-run renaming")

        print(f"  exit code: {result.returncode}")
        if result.returncode != 0:
            failures += 1
            if args.fail_fast:
                break

    print(f"\nCompleted with {failures} failing run(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
