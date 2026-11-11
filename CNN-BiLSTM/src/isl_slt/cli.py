"""Documented command dispatcher. Parsing and planning never import or execute project ML code."""

import argparse
import json
from pathlib import Path

from isl_slt.config import Config, load_config
from isl_slt.utils.io import read_jsonl, write_json, write_jsonl


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="isl_slt", description="Auditable continuous ISL translation")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "env-check", "audit", "split", "check-leakage", "prepare-features", "prepare-model",
                 "prepare-rgb", "prepare-stats", "verify-cache", "train", "overfit", "evaluate", "predict", "diagnostics", "benchmark", "make-source-map"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--dry-run", action="store_true", help="Print resolved plan without loading data/models")
        if name in ("verify-cache", "train", "overfit", "evaluate", "diagnostics"):
            sub.add_argument("--verify-assets", action=argparse.BooleanOptionalAction, default=None)
        if name == "make-source-map":
            sub.add_argument("--out")
            sub.add_argument("--dataset", choices=["target", "pretrain"], default="target")
            sub.add_argument("--promote", action="store_true")
            sub.add_argument("--evidence")
            sub.add_argument("--approve-def", action="store_true")
        if name == "audit":
            sub.add_argument("--allow-missing-poses", action="store_true", help="Write a draft envelope, never a frozen manifest")
        if name == "prepare-model":
            sub.add_argument("--allow-network", action="store_true")
        if name in ("train", "overfit"):
            sub.add_argument("--resume")
            sub.add_argument("--seed", type=int)
            sub.add_argument("--allow-code-change", action="store_true")
            sub.add_argument("--allow-environment-change", action="store_true")
        if name == "overfit":
            sub.add_argument("--uids", required=True)
        if name in ("evaluate", "predict", "diagnostics", "benchmark"):
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--output", required=True)
        if name in ("evaluate", "diagnostics"):
            sub.add_argument("--partition", choices=["validation", "test", "gold"], default="test")
        if name in ("evaluate", "predict", "benchmark", "diagnostics"):
            sub.add_argument("--decoding")
        if name in ("predict", "benchmark"):
            sub.add_argument("--inputs", required=True, help="JSONL VisualInput records; reference fields are rejected")
    average = commands.add_parser("average")
    average.add_argument("--checkpoints", nargs="+", required=True)
    average.add_argument("--output", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--reports", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--first", required=True)
    bootstrap.add_argument("--second", required=True)
    bootstrap.add_argument("--samples", type=int, default=1000)
    bootstrap.add_argument("--seed", type=int, default=2026)
    bootstrap.add_argument("--output", required=True)
    estimate = commands.add_parser("estimate")
    estimate.add_argument("--examples", type=int, required=True)
    estimate.add_argument("--epochs", type=int, required=True)
    estimate.add_argument("--measured-examples-per-second", type=float)
    estimate.add_argument("--validation-seconds-per-epoch", type=float)
    estimate.add_argument("--run-summary", help="Measured completed-epoch stage summary JSON")
    estimate.add_argument("--stage", help="Explicit stage in --run-summary")
    estimate.add_argument("--output")
    estimate.add_argument("--hourly-price", type=float)
    return result


def dispatch(args) -> dict | list:
    command = args.command
    if hasattr(args, "config"):
        config = load_config(args.config)
        raw = config.model_dump(mode="json")
        if getattr(args, "verify_assets", None) is not None:
            raw["data"]["verify_assets"] = args.verify_assets
        if getattr(args, "seed", None) is not None:
            raw["training"]["seed"] = args.seed
            raw["run_dir"] = str(Path(raw["run_dir"]) / f"seed-{args.seed}")
        if command == "overfit":
            raw["training"]["overfit_uids"] = args.uids
        config = Config.model_validate(raw)
        if command == "plan" or args.dry_run:
            from isl_slt.workflows import planned_experiment
            return {"command": command, **planned_experiment(config)}
    if command == "verify-cache":
        from isl_slt.workflows import verify_cache
        return verify_cache(config, args.verify_assets is not False)
    if command == "prepare-stats":
        from isl_slt.features.statistics import prepare_stats
        return prepare_stats(config)
    if command == "make-source-map":
        from isl_slt.workflows import make_source_map
        return make_source_map(config, args.out, args.dataset, promote=args.promote,
                               evidence=args.evidence, approve_def=args.approve_def)
    if command == "env-check":
        from isl_slt.utils.runtime import environment_report
        report = environment_report(config.runtime)
        if not report["configured_precision_ok"]:
            print(json.dumps(report, indent=2))
            raise RuntimeError("Configured precision failed recurrent/T5 environment probes; inspect report before training")
        return report
    if command == "audit":
        from isl_slt.workflows import audit
        return audit(config, args.allow_missing_poses)
    if command in ("split", "prepare-features", "prepare-rgb"):
        from isl_slt import workflows
        return getattr(workflows, command.replace("-", "_"))(config)
    if command == "check-leakage":
        from isl_slt.data.splits import checked_splits
        _, metadata = checked_splits(config.data)
        return {"ok": True, "split_metadata": metadata}
    if command == "prepare-model":
        from isl_slt.workflows import prepare_model
        return prepare_model(config, args.allow_network)
    if command in ("train", "overfit"):
        from isl_slt.training.engine import run_training
        return run_training(config, args.resume, args.allow_code_change, args.allow_environment_change)
    if command == "evaluate":
        from isl_slt.workflows import evaluate
        report = evaluate(config, args.checkpoint, args.partition, args.output, args.decoding)
        if report["metrics"]["unavailable"]:
            raise RuntimeError(f"Predictions/report saved to {args.output}; prepare missing requested metric resources: "
                               f"{sorted(report['metrics']['unavailable'])}")
        return report
    if command == "diagnostics":
        from isl_slt.workflows import diagnostics
        return diagnostics(config, args.checkpoint, args.partition, args.output, args.decoding)
    if command in ("predict", "benchmark"):
        from isl_slt.inference import Predictor, VisualInput
        predictor = Predictor.from_checkpoint(args.checkpoint, config.runtime, args.decoding, config.generation)
        inputs = [VisualInput.model_validate(row) for row in read_jsonl(args.inputs)]
        if command == "predict":
            predictions = predictor.predict(inputs)
            write_jsonl(args.output, predictions)
            return {"output": args.output, "count": len(predictions)}
        from isl_slt.evaluation.analysis import benchmark
        report = benchmark(predictor, inputs, config.evaluation.warmup_iterations, config.evaluation.benchmark_iterations)
    elif command == "average":
        from isl_slt.training.checkpoint import average_checkpoints
        return average_checkpoints(args.checkpoints, args.output)
    elif command == "summarize":
        from isl_slt.evaluation.analysis import aggregate_runs
        report = aggregate_runs(args.reports)
    elif command == "bootstrap":
        from isl_slt.evaluation.analysis import paired_source_bootstrap
        report = paired_source_bootstrap(read_jsonl(args.first), read_jsonl(args.second), args.samples, args.seed)
    elif command == "estimate":
        from isl_slt.evaluation.analysis import estimate_budget
        throughput, validation = args.measured_examples_per_second, args.validation_seconds_per_epoch
        if args.run_summary:
            from isl_slt.utils.io import read_json
            if not args.stage or throughput is not None or validation is not None:
                raise ValueError("--run-summary requires --stage and cannot mix manual throughput/validation inputs")
            stage = read_json(args.run_summary).get("stages", {}).get(args.stage)
            if not stage or not stage.get("examples_per_second"):
                raise ValueError("Stage has no measured completed-epoch throughput")
            throughput, validation = stage["examples_per_second"], stage["validation_seconds_per_epoch"]
        elif args.stage or throughput is None or validation is None:
            raise ValueError("Supply both measured values, or --run-summary with --stage")
        report = estimate_budget(args.examples, args.epochs, throughput, validation, args.hourly_price)
        report["measurement_source"] = {"summary": args.run_summary, "stage": args.stage}
        if args.output:
            write_json(args.output, report)
        return report
    else:
        raise ValueError(f"Unknown command {command}")
    write_json(args.output, report)
    return report


def main(argv=None) -> None:
    arguments = parser().parse_args(argv)
    try:
        result = dispatch(arguments)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        raise SystemExit(f"isl_slt: {error}") from error
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
