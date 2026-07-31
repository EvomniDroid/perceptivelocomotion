"""Expand a trained B2RM velocity checkpoint for the handoff observations."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


OLD_OBSERVATION_DIM = 72
NEW_OBSERVATION_DIM = 74
EXPECTED_EXPANDED_TENSORS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a 72-D B2RM velocity policy into the 74-D handoff policy. "
            "The two new handoff-state input columns are initialized to zero."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Source model_*.pt checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Destination model_<source_iter>.pt checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination if it already exists.")
    return parser.parse_args()


def migrate_checkpoint(source: dict) -> tuple[dict, list[str]]:
    if "model_state_dict" not in source:
        raise KeyError("Source checkpoint has no 'model_state_dict'.")

    migrated_model = OrderedDict()
    expanded_keys: list[str] = []

    for name, value in source["model_state_dict"].items():
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == OLD_OBSERVATION_DIM:
            expanded = value.new_zeros((value.shape[0], NEW_OBSERVATION_DIM))
            expanded[:, :OLD_OBSERVATION_DIM].copy_(value)
            migrated_model[name] = expanded
            expanded_keys.append(name)
        else:
            migrated_model[name] = value

    if len(expanded_keys) != EXPECTED_EXPANDED_TENSORS:
        raise RuntimeError(
            f"Expected to expand {EXPECTED_EXPANDED_TENSORS} tensors, but expanded {len(expanded_keys)}: "
            f"{expanded_keys}"
        )

    # The old Adam moments have 72-D shapes and cannot be loaded into the 74-D policy.
    migrated = {
        key: value
        for key, value in source.items()
        if key not in {"model_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict"}
    }
    migrated["model_state_dict"] = migrated_model
    migrated["iter"] = source.get("iter", 0)
    migrated["infos"] = {
        "warm_start": "b2rm_velocity_72d_to_handoff_74d",
        "source_iteration": source.get("iter"),
        "new_input_columns": ["handoff_action_alpha", "handoff_gain_alpha"],
    }
    return migrated, expanded_keys


def main() -> None:
    args = parse_args()
    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    migrated, expanded_keys = migrate_checkpoint(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output_path)

    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Iteration: {source.get('iter')} -> {migrated['iter']}")
    print("Expanded input tensors:")
    for key in expanded_keys:
        print(f"  - {key}: {OLD_OBSERVATION_DIM} -> {NEW_OBSERVATION_DIM}")
    print("Optimizer state: reset")


if __name__ == "__main__":
    main()
