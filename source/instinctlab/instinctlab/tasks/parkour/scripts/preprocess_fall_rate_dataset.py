"""Preprocess B2RM fall-rate eval records into a smaller supervised dataset.

The eval script records many frames per episode. For some terrains, only the
first few frames show the useful upcoming obstacle before the robot reaches it.
This script filters frames per terrain/episode and copies the selected images
plus labels into a processed dataset directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from typing import Any


DEFAULT_KEEP_FRAMES = {
    "pit_crater": 2,
    "raised_mound": 3,
}


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_keep_frames(items: list[str]) -> dict[str, int]:
    rules = dict(DEFAULT_KEEP_FRAMES)
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid keep rule {item!r}; expected terrain=N.")
        terrain, count_text = item.split("=", 1)
        terrain = terrain.strip()
        if not terrain:
            raise ValueError(f"Invalid keep rule {item!r}; terrain name is empty.")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"Invalid keep rule {item!r}; N must be >= 0.")
        rules[terrain] = count
    return rules


def _copy_file(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported copy mode: {mode}")


def _write_label(path: str, metadata: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def _episode_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    return (
        meta.get("terrain"),
        int(meta.get("row", -1)),
        int(meta.get("col", -1)),
        int(meta.get("env_id", -1)),
        int(meta.get("episode_id", -1)),
        meta.get("modality"),
    )


def _load_metadata(record_dir: str) -> list[dict[str, Any]]:
    metadata_path = os.path.join(record_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

    rows: list[dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {metadata_path}:{line_no}: {exc}") from exc
    return rows


def _relative_output_path(meta: dict[str, Any], source_rel: str) -> str:
    modality = str(meta.get("modality", "unknown"))
    terrain = str(meta.get("terrain", "unknown"))
    row = int(meta.get("row", -1))
    col = int(meta.get("col", -1))
    env_id = int(meta.get("env_id", -1))
    episode_id = int(meta.get("episode_id", -1))
    step = int(meta.get("step", -1))
    ext = os.path.splitext(source_rel)[1]
    stem = f"step{step:08d}_ep{episode_id:05d}_env{env_id:03d}"
    return os.path.join(modality, terrain, f"row{row:02d}_col{col:02d}", f"env{env_id:03d}", stem + ext)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter B2RM eval frames for supervised fall-rate training.")
    parser.add_argument("--record_dir", required=True, help="Eval output directory containing metadata.jsonl.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Processed output directory. Defaults to <record_dir>/processed/<timestamp>.",
    )
    parser.add_argument(
        "--modalities",
        default="rgb_depth,raycaster_depth,rgb",
        help="Comma-separated modalities to keep.",
    )
    parser.add_argument(
        "--keep_frames",
        action="append",
        default=[],
        help="Per-terrain first-frame rule, e.g. --keep_frames pit_crater=2 --keep_frames raised_mound=3.",
    )
    parser.add_argument(
        "--default_keep_frames",
        type=int,
        default=0,
        help="Frames per episode to keep for terrains not listed in --keep_frames. 0 skips them.",
    )
    parser.add_argument(
        "--include_incomplete",
        action="store_true",
        default=False,
        help="Include frames whose episode did not finish before eval stopped.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=("copy", "hardlink", "symlink"),
        default="hardlink",
        help="How to materialize selected files in the processed dataset.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help="Only print summary; do not write files.")
    args = parser.parse_args()

    record_dir = os.path.abspath(args.record_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(args.output_dir or os.path.join(record_dir, "processed", run_id))
    modalities = set(_parse_csv_list(args.modalities))
    keep_rules = _parse_keep_frames(args.keep_frames)

    if args.default_keep_frames < 0:
        raise ValueError("--default_keep_frames must be >= 0.")

    rows = _load_metadata(record_dir)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for meta in rows:
        if meta.get("modality") not in modalities:
            continue
        if not args.include_incomplete and not bool(meta.get("episode_complete", False)):
            continue
        if not meta.get("png"):
            continue
        groups[_episode_key(meta)].append(meta)

    selected: list[dict[str, Any]] = []
    skipped_by_terrain: Counter[str] = Counter()
    selected_by_terrain: Counter[str] = Counter()
    selected_by_modality: Counter[str] = Counter()

    for _key, frames in groups.items():
        frames.sort(key=lambda item: int(item.get("step", -1)))
        terrain = str(frames[0].get("terrain", "unknown"))
        keep_count = keep_rules.get(terrain, args.default_keep_frames)
        if keep_count <= 0:
            skipped_by_terrain[terrain] += len(frames)
            continue
        kept = frames[:keep_count]
        selected.extend(kept)
        selected_by_terrain[terrain] += len(kept)
        for meta in kept:
            selected_by_modality[str(meta.get("modality", "unknown"))] += 1

    print(f"[PREPROCESS] record_dir={record_dir}")
    print(f"[PREPROCESS] output_dir={output_dir}")
    print(f"[PREPROCESS] modalities={sorted(modalities)}")
    print(f"[PREPROCESS] keep_rules={keep_rules}, default_keep_frames={args.default_keep_frames}")
    print(f"[PREPROCESS] input frames={len(rows)}, grouped episodes/modalities={len(groups)}, selected frames={len(selected)}")
    print(f"[PREPROCESS] selected_by_terrain={dict(sorted(selected_by_terrain.items()))}")
    print(f"[PREPROCESS] selected_by_modality={dict(sorted(selected_by_modality.items()))}")
    if skipped_by_terrain:
        print(f"[PREPROCESS] skipped_by_terrain={dict(sorted(skipped_by_terrain.items()))}")

    if args.dry_run:
        return

    os.makedirs(output_dir, exist_ok=True)
    manifest_jsonl = os.path.join(output_dir, "manifest.jsonl")
    manifest_csv = os.path.join(output_dir, "manifest.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    csv_fields = [
        "png",
        "label_txt",
        "source_png",
        "source_label_txt",
        "modality",
        "terrain",
        "row",
        "col",
        "env_id",
        "episode_id",
        "step",
        "episode_fell",
        "episode_timeout",
        "termination_reason",
        "cell_fall_rate_after",
        "depth_min",
        "depth_max",
        "rgb_min",
        "rgb_max",
    ]

    with open(manifest_jsonl, "w", encoding="utf-8") as jf, open(
        manifest_csv, "w", newline="", encoding="utf-8"
    ) as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()

        for meta in selected:
            source_png_rel = meta["png"]
            source_png_abs = os.path.join(record_dir, source_png_rel)
            if not os.path.exists(source_png_abs):
                print(f"[PREPROCESS][WARN] missing image: {source_png_abs}")
                continue

            out_png_rel = _relative_output_path(meta, source_png_rel)
            out_png_abs = os.path.join(output_dir, out_png_rel)
            _copy_file(source_png_abs, out_png_abs, args.copy_mode)

            out_meta = dict(meta)
            out_meta["source_png"] = source_png_rel
            out_meta["png"] = out_png_rel

            source_label_rel = meta.get("label_txt") or os.path.splitext(source_png_rel)[0] + ".txt"
            source_label_abs = os.path.join(record_dir, source_label_rel)
            out_label_rel = os.path.splitext(out_png_rel)[0] + ".txt"
            out_label_abs = os.path.join(output_dir, out_label_rel)
            out_meta["source_label_txt"] = source_label_rel
            out_meta["label_txt"] = out_label_rel
            if os.path.exists(source_label_abs):
                _copy_file(source_label_abs, out_label_abs, args.copy_mode)
            else:
                _write_label(out_label_abs, out_meta)

            for companion_key in ("npy", "u16_png"):
                source_rel = meta.get(companion_key)
                if not source_rel:
                    continue
                source_abs = os.path.join(record_dir, source_rel)
                if not os.path.exists(source_abs):
                    continue
                out_rel = _relative_output_path(meta, source_rel)
                _copy_file(source_abs, os.path.join(output_dir, out_rel), args.copy_mode)
                out_meta[f"source_{companion_key}"] = source_rel
                out_meta[companion_key] = out_rel

            jf.write(json.dumps(out_meta, ensure_ascii=False) + "\n")
            writer.writerow({field: out_meta.get(field, "") for field in csv_fields})

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "record_dir": record_dir,
                "output_dir": output_dir,
                "modalities": sorted(modalities),
                "keep_rules": keep_rules,
                "default_keep_frames": args.default_keep_frames,
                "include_incomplete": args.include_incomplete,
                "copy_mode": args.copy_mode,
                "input_frames": len(rows),
                "grouped_episode_modalities": len(groups),
                "selected_frames": len(selected),
                "selected_by_terrain": dict(sorted(selected_by_terrain.items())),
                "selected_by_modality": dict(sorted(selected_by_modality.items())),
                "skipped_by_terrain": dict(sorted(skipped_by_terrain.items())),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[PREPROCESS] wrote {manifest_jsonl}")
    print(f"[PREPROCESS] wrote {manifest_csv}")
    print(f"[PREPROCESS] wrote {summary_path}")


if __name__ == "__main__":
    main()
