"""Check a trained visual model with observed pixels versus explicitly masked pixels."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pcat_slt.data import SignDataset, make_collator, model_inputs, move_batch
from pcat_slt.runtime import load_checkpoint, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="reports/visual_intervention.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    if cfg.model.visual_backend == "none":
        raise ValueError("visual checkpoint required")
    dataset = SignDataset(args.manifest or cfg.data.test, cfg.model, cfg.data, model.codec)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, collate_fn=make_collator(model.codec))
    rows = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            eligible = model.frame.eligible_frames(batch["pose"], batch["confidence"], batch["lengths"]).any(1)
            if not eligible.all():
                raise ValueError("masked-RGB diagnostic requires an anchored pose for every example")
            observed = model.generate(**model_inputs(batch, decoder=False))
            batch["rgb_mask"] = torch.zeros_like(batch["rgb_mask"])
            masked = model.generate(**model_inputs(batch, decoder=False))
            for uid, reference, a, b in zip(batch["ids"], batch["texts"], observed, masked):
                rows.append({"id": uid, "reference": reference,
                             "observed": model.codec.decode(a), "masked": model.codec.decode(b)})
    report = {"samples": len(rows),
              "observed_exact_match": sum(r["observed"] == r["reference"] for r in rows) / len(rows),
              "masked_exact_match": sum(r["masked"] == r["reference"] for r in rows) / len(rows),
              "rows": rows,
              "interpretation": "Within-checkpoint observation ablation; not an official SLT score or linguistic grounding proof."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
