"""Paired, reproducible perturbations. They test input dependence, not semantic faithfulness."""

import random

import torch


def perturb_items(items: list[dict], kind: str, seed: int) -> tuple[list[dict], list[dict]]:
    if kind not in ("real", "blank", "incorrect", "shuffle"):
        raise ValueError(f"Unknown visual diagnostic: {kind}")
    if kind == "incorrect" and len(items) < 2:
        raise ValueError("Incorrect-example assignment requires at least two examples")
    rng = random.Random(seed)
    order = list(range(len(items)))
    if kind == "incorrect":
        rng.shuffle(order)
        mapping = {order[index]: order[(index + 1) % len(order)] for index in range(len(order))}
    else:
        mapping = dict(enumerate(order))
    result, flags = [], []
    generator = torch.Generator().manual_seed(seed)
    for index, original in enumerate(items):
        donor = items[mapping[index]]
        changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in donor.items()}
        changed["record"] = original["record"]
        flag = {"uid": original["record"].uid, "kind": kind, "seed": seed}
        if kind == "incorrect":
            flag.update(donor_uid=donor["record"].uid,
                        same_source=donor["record"].source_id == original["record"].source_id)
        elif kind == "blank":
            changed["features"].zero_()
            if "images" in changed:
                shape = (*changed["images"].shape[:2], 512)
                changed.pop("images")
                changed["rgb"] = torch.zeros(shape)
            elif "rgb" in changed:
                changed["rgb"].zero_()
        elif kind == "shuffle":
            permutation = torch.randperm(len(changed["features"]), generator=generator)
            for key in ("features", "rgb", "images", "rgb_mask"):
                if key in changed:
                    changed[key] = changed[key][permutation]
        result.append(changed)
        flags.append(flag)
    return result, flags
