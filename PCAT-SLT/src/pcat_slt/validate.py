"""Validate every sample, geometry, split leakage and optional signer independence."""
import argparse
import hashlib
import json
from collections import Counter
import torch
from .config import load_config
from .data import SignDataset, assert_disjoint
from .geometry import BodyFrame


def validate(cfg, signer_independent=False, codec=None):
    if codec is None:
        from .text import load_text_codec
        codec = load_text_codec(cfg.model)
    datasets = {name: SignDataset(getattr(cfg.data, name), cfg.model, cfg.data, codec)
                for name in ("train", "val", "test")}
    assert_disjoint(datasets)
    seen_content, signers, report = {}, {}, {}
    seen_rgb = {}
    frame = BodyFrame(cfg.model.anchors, cfg.model.anchor_min_confidence, cfg.model.anchor_policy)
    for split, dataset in datasets.items():
        frames, missing, counts = [], 0, Counter()
        fallback, unanchored, known_sources = 0, 0, set()
        for index in range(len(dataset)):
            sample = dataset[index]
            eligible = frame.eligible_frames(sample["pose"][None], sample["confidence"][None])
            if eligible.any():
                frame(sample["pose"][None], sample["confidence"][None])
                fallback += int(eligible.long().argmax(1)[0] > 0)
            elif cfg.model.allow_unanchored_rgb and "rgb_mask" in sample and sample["rgb_mask"].any():
                unanchored += 1
            else:
                raise ValueError("sample has neither anchored pose nor observed RGB")
            source = dataset.records[index].get("source_video")
            if source:
                known_sources.add(source)
            content = hashlib.sha256()
            for name in ("pose", "confidence", "timestamps"):
                content.update(sample[name].numpy().tobytes())
            digest = content.hexdigest()
            if digest in seen_content:
                raise ValueError(f"duplicate decoded pose content: {seen_content[digest]} and {sample['id']}")
            seen_content[digest] = sample["id"]
            if "rgb" in sample:
                pixels = hashlib.sha256(sample["rgb"].numpy().tobytes()
                                        + sample["rgb_mask"].numpy().tobytes()).hexdigest()
                if pixels in seen_rgb and seen_rgb[pixels][0] != split:
                    raise ValueError(f"duplicate RGB content across {seen_rgb[pixels][0]} and {split}")
                seen_rgb[pixels] = (split, sample["id"])
            signer = dataset.records[index].get("signer")
            if signer_independent and (not isinstance(signer, str) or not signer):
                raise ValueError("signer-independent validation requires nonempty signer strings on every row")
            if signer:
                counts[str(signer)] += 1
            frames.append(len(sample["pose"]))
            missing += int(sample["confidence"].eq(0).sum())
        signers[split] = set(counts)
        report[split] = {"samples": len(dataset), "frames_min": min(frames), "frames_max": max(frames),
                         "missing_joint_observations": missing, "known_signers": len(counts),
                         "anchor_fallback_samples": fallback, "known_source_videos": len(known_sources)}
        report[split]["unanchored_rgb_samples"] = unanchored
    overlap = {f"{a}/{b}": sorted(signers[a] & signers[b]) for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))}
    if signer_independent and any(overlap.values()):
        raise ValueError(f"signer overlap: {overlap}")
    report["signer_overlap"] = overlap
    report["content_hash_scope"] = "exact decoded arrays; near-duplicate and source-video leakage require provenance review"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--signer-independent", action="store_true")
    args = parser.parse_args()
    with torch.no_grad():
        print(json.dumps(validate(load_config(args.config), args.signer_independent), indent=2))


if __name__ == "__main__":
    main()
