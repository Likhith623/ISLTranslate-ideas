"""Autoregressive corpus evaluation; checkpoint determines architecture and tokenizer."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import SignDataset, make_collator, model_inputs, move_batch
from .runtime import load_checkpoint, resolve_device, file_sha256
from .metrics import translation_metrics


def evaluate(checkpoint, manifest=None, device="auto", output="evaluation", max_new_tokens=None):
    device = resolve_device(device)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    model.eval()
    manifest = manifest or cfg.data.test
    dataset = SignDataset(manifest, cfg.model, cfg.data, model.codec)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=cfg.data.workers, collate_fn=make_collator(model.codec))
    tokenizer = model.codec
    hypotheses, references, rows = [], [], []
    invalid_utf8 = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            generated = model.generate(**model_inputs(batch, decoder=False), max_new_tokens=max_new_tokens)
            for sample_id, text, tokens in zip(batch["ids"], batch["texts"], generated.cpu().tolist()):
                hypothesis = tokenizer.decode(tokens)
                try:
                    tokenizer.decode(tokens, errors="strict")
                except UnicodeDecodeError:
                    invalid_utf8 += 1
                hypotheses.append(hypothesis)
                references.append(text)
                rows.append({"id": sample_id, "reference": text, "hypothesis": hypothesis,
                             "ended_with_eos": tokenizer.eos_id in tokens})
    metrics = translation_metrics(hypotheses, references)
    metrics.update(checkpoint_sha256=file_sha256(checkpoint), manifest_sha256=file_sha256(manifest),
                   generation=f"{cfg.model.text_backend}; beams={cfg.model.generation_beams}",
                   max_new_tokens=max_new_tokens or cfg.model.max_target_length,
                   eos_fraction=sum(r["ended_with_eos"] for r in rows) / len(rows))
    metrics["invalid_utf8_fraction"] = invalid_utf8 / len(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output", default="evaluation")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.manifest, args.device, args.output, args.max_new_tokens)


if __name__ == "__main__":
    main()
