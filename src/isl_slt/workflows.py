"""CLI workflows composed from production functions; heavy imports occur inside explicit operations."""

from pathlib import Path

from isl_slt.config import Config
from isl_slt.utils.io import file_hash, fingerprint, read_json, write_json, write_jsonl
from isl_slt.data.context import SourceIndex
from isl_slt.features.cache import SignatureSession
from isl_slt.evaluation.selection import load_decoding_selection


def planned_experiment(config: Config) -> dict:
    return {"experiment": config.experiment, "run_dir": config.run_dir, "execution": "planned only",
            "config": config.model_dump(mode="json"), "stages": [stage.model_dump() for stage in config.training.stages],
            "assets": {name: {"path": path, "exists": Path(path).exists()} for name, path in {
                "annotations": config.data.target.annotations, "poses": config.data.target.pose_root,
                "splits": str(Path(config.data.split_dir) / "splits.json"), "t5": config.model.t5_dir}.items()},
            "network": "Only explicit prepare-model --allow-network may download assets",
            "numerical_validation": "not performed by planning"}


def make_source_map(config: Config, output: str | None = None, dataset: str = "target", *,
                    promote: bool = False, evidence: str | None = None, approve_def: bool = False) -> dict:
    import csv
    from collections import Counter
    from isl_slt.data.manifests import mapped_columns, read_csv, source_candidate
    from isl_slt.utils.io import atomic_open
    adapter = config.data.target if dataset == "target" else config.data.pretrain
    if adapter is None:
        raise ValueError("No pretrain adapter is configured")
    if (evidence or approve_def) and not promote:
        raise ValueError("--evidence and --approve-def require --promote")
    if promote:
        if not evidence or not evidence.strip():
            raise ValueError("Promotion requires nonempty --evidence supporting source identities")
        if adapter.dataset_revision.strip().lower() in {"required", "main", "master", "latest", "head", "", "unverified-extraction-request"}:
            raise ValueError("Pin the exact dataset revision before source promotion")
        if not adapter.source_mapping:
            raise ValueError("Configure the authoritative source_mapping destination")
        if Path(adapter.source_mapping).exists():
            raise ValueError("Promotion refuses to overwrite an existing source map")
    headers, raw = read_csv(adapter.annotations)
    columns = mapped_columns(headers, {"uid": adapter.columns["uid"]})
    destination = Path(output) if output else Path(adapter.source_mapping or f"data/metadata/{dataset}_source_mapping.csv").with_suffix(".candidates.csv")
    if destination.resolve() == Path(adapter.annotations).resolve() or (
        adapter.source_mapping and destination.resolve() == Path(adapter.source_mapping).resolve()
    ):
        raise ValueError("Write candidates separately from raw annotations and authoritative source mappings")
    rows, rules, seen, unresolved = [], [], {}, []
    for row in raw:
        uid = row[columns["uid"]]
        if uid in seen:
            if seen[uid] != row:
                raise ValueError(f"Conflicting duplicate UID {uid!r}")
            continue
        seen[uid] = row
        match, rule, pattern = source_candidate(uid, adapter.candidate_uid_pattern or adapter.candidate_uid_patterns)
        rows.append({"uid": uid, "source_id": match["source"] if match else "",
                     "verified": "false", "provenance": f"candidate:{rule}"})
        rules.append({"uid": uid, "rule": rule, "pattern": pattern})
        if match is None:
            unresolved.append(uid)
    with atomic_open(destination, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uid", "source_id", "verified", "provenance"])
        writer.writeheader()
        writer.writerows(rows)
    report = {"dataset": adapter.dataset_name, "revision": adapter.dataset_revision,
              "annotations_sha256": file_hash(adapter.annotations), "output": str(destination),
              "count": len(rows), "rules": dict(Counter(row["rule"] for row in rules)),
              "candidates": rules, "unresolved": unresolved,
              "verification": "All rows remain unverified; review cross-dataset identity and record supporting evidence"}
    write_json(destination.with_suffix(".audit.json"), report)
    if unresolved:
        raise ValueError(f"Unresolved UIDs (candidate report saved): {unresolved}")
    if promote:
        from isl_slt.config import Adapter
        allowed_patterns = set(Adapter().candidate_uid_patterns)
        if any(row["rule"] not in {"sentence-suffix", "def-tag"} or row["pattern"] not in allowed_patterns for row in rules):
            raise ValueError("Custom UID patterns require a separately reviewed authoritative map")
        if any(row["rule"] == "def-tag" for row in rules) and not approve_def:
            raise ValueError("DEF source identities require explicit --approve-def; no source map promoted")
        provenance = {"evidence": evidence.strip(), "dataset": adapter.dataset_name,
                      "dataset_revision": adapter.dataset_revision, "annotations_sha256": report["annotations_sha256"],
                      "def_approved": approve_def, "basis": "explicitly approved dataset UID convention"}
        import json
        promoted = [{**row, "verified": "true", "provenance": json.dumps(
            {**provenance, "rule": rule["rule"], "pattern": rule["pattern"]}, sort_keys=True)}
            for row, rule in zip(rows, rules)]
        with atomic_open(adapter.source_mapping, "w", replace=False) as stream:
            writer = csv.DictWriter(stream, fieldnames=["uid", "source_id", "verified", "provenance"])
            writer.writeheader()
            writer.writerows(promoted)
        report["promotion"] = {"destination": adapter.source_mapping, **provenance}
        write_json(destination.with_suffix(".audit.json"), report)
    return report


def audit(config: Config, allow_missing_poses: bool = False) -> dict:
    from isl_slt.data.dataset import raw_sequence
    from isl_slt.data.manifests import adapt_csv, attach_interpreter, audit_records, save_manifest
    from isl_slt.data.pose import read_pose
    from isl_slt.features.pose import prepare_pose
    reports, pending, failures = {}, [], []
    for kind, adapter, destination in (("target", config.data.target, config.data.manifest),
                                       ("pretrain", config.data.pretrain, config.data.pretrain_manifest)):
        if adapter is None:
            continue
        adapter_audit = {}
        records = adapt_csv(adapter, adapter_audit)
        if kind == "target" and config.data.interpreter_csv:
            records = attach_interpreter(records, config.data.interpreter_csv, config.data.interpreter_columns)
        locked = []
        for record in records:
            hashes = {name: file_hash(path) for name, path in {
                "pose": record.pose_path, "video": record.video_path, "timestamps": record.timestamps_path
            }.items() if path and Path(path).is_file()}
            if any(hashes[name] != value for name, value in record.asset_hashes.items() if name in hashes):
                raise ValueError(f"{record.uid}: supplied asset checksums conflict with local files")
            duplicates = sorted(set(record.duplicate_fingerprints) | {
                f"{name}-file-sha256:{value}" for name, value in hashes.items() if name in ("pose", "video")})
            locked.append(record.model_copy(update={"asset_hashes": hashes, "duplicate_fingerprints": duplicates}))
        records = locked
        report = {**audit_records(records), **adapter_audit,
                  "annotations_sha256": file_hash(adapter.annotations), "dataset_revision": adapter.dataset_revision}
        pose_audits = []
        for record in records:
            if not Path(record.pose_path).is_file():
                continue
            try:
                sequence = read_pose(record, config.features)
                prepared = prepare_pose(sequence, config.features, config.context.enabled)
                pose_audits.append({"uid": record.uid, "frames": len(sequence.times), "fps": sequence.metadata["fps"],
                    "missing_selected_landmarks": int((~prepared.observed).sum()),
                    "total_selected_landmarks": prepared.observed.size, "components": sequence.metadata["components"],
                    "input_dim": prepared.values.shape[1], "rgb_transform_present": record.image_transform is not None})
            except (ValueError, OSError) as error:
                pose_audits.append({"uid": record.uid, "error": str(error)})
        report["poses"] = pose_audits
        report["context_requested"] = config.context.enabled
        report["rgb_requested"] = config.rgb.mode != "off"
        if config.rgb.mode != "off":
            report["rgb_unavailable"] = [record.uid for record in records if not record.video_path or
                not Path(record.video_path).is_file() or record.image_transform is None]
        if config.context.enabled:
            context_errors = []
            indexed = SourceIndex(records)
            for record in records:
                try:
                    raw_sequence(record, indexed, config)
                except (ValueError, OSError) as error:
                    context_errors.append({"uid": record.uid, "error": str(error)})
            report["context_errors"] = context_errors
        errors = {
            "missing_pose": report["missing_pose"],
            "invalid_pose": [row for row in pose_audits if "error" in row],
            "invalid_targets": report["invalid_targets"],
            "rgb_unavailable": report.get("rgb_unavailable", []),
            "context_errors": report.get("context_errors", []),
        }
        report["errors"] = {name: rows for name, rows in errors.items() if rows}
        report["status"] = "draft" if allow_missing_poses else "failed" if report["errors"] else "validated"
        write_json(Path(destination).with_suffix(".audit.json"), report)
        reports[kind] = report
        pending.append((destination, records))
        if report["errors"]:
            failures.append(kind)
    if allow_missing_poses:
        for destination, records in pending:
            draft = Path(destination).with_suffix(".draft.json")
            write_json(draft, {"format": "isl-slt-draft-v1", "frozen": False,
                              "records": [r.model_dump(mode="json") for r in records]})
        return reports
    if failures:
        raise ValueError(f"Audit failed for {failures}; inspect .audit.json reports. No manifests were frozen. "
                         "Use --allow-missing-poses only for a non-training draft.")
    # Preflight existing destinations before publishing any member of a multi-dataset audit.
    from isl_slt.data.manifests import load_manifest, manifest_hash
    for destination, records in pending:
        if Path(destination).exists() and manifest_hash(load_manifest(destination)) != manifest_hash(records):
            raise ValueError(f"Refusing to replace immutable manifest: {destination}; use a new revision/path")
    for destination, records in pending:
        save_manifest(destination, records)
    return reports


def split(config: Config) -> dict:
    from isl_slt.data.manifests import load_manifest
    from isl_slt.data.splits import export_splits, filter_pretraining, make_splits
    partitions = make_splits(load_manifest(config.data.manifest), config.data)
    contamination = None
    if config.data.pretrain is not None:
        kept, contamination = filter_pretraining(load_manifest(config.data.pretrain_manifest), partitions)
        if not kept:
            raise ValueError("No eligible additional iSign examples remain after filtering")
        partitions["pretrain"] = kept
    return export_splits(partitions, config.data, contamination)


def prepare_features(config: Config) -> dict:
    from isl_slt.data.dataset import dependencies_for, raw_sequence
    from isl_slt.data.splits import checked_splits
    from isl_slt.features.cache import preprocessing_fingerprint, write_features
    from isl_slt.features.pose import prepare_pose
    partitions, _ = checked_splits(config.data)
    results = {}
    signatures = SignatureSession()
    for name, records in partitions.items():
        indexed = SourceIndex(records)
        counts = []
        for record in records:
            sequence = raw_sequence(record, indexed, config)
            prepared = prepare_pose(sequence, config.features, config.context.enabled)
            metadata = write_features(config, record, prepared, dependencies_for(record, indexed, config), signatures)
            counts.append({"uid": record.uid, "frames": metadata["frames"], "input_dim": metadata["input_dim"]})
        results[name] = counts
    report = {"preprocessing_fingerprint": preprocessing_fingerprint(config), "partitions": results}
    write_json(Path(config.data.feature_dir) / preprocessing_fingerprint(config) / "index.json", report)
    return {"preprocessing_fingerprint": report["preprocessing_fingerprint"],
            "counts": {name: len(rows) for name, rows in results.items()}}


def verify_cache(config: Config, verify_assets: bool = True) -> dict:
    from isl_slt.data.dataset import dependencies_for
    from isl_slt.data.splits import checked_splits
    from isl_slt.features.cache import read_features
    partitions, _ = checked_splits(config.data)
    counts = {}
    signatures = SignatureSession()
    for name, records in partitions.items():
        indexed = SourceIndex(records)
        for record in records:
            dependencies = dependencies_for(record, indexed, config)
            prepared = read_features(config, record, dependencies, verify_assets=verify_assets, signatures=signatures)
            if config.rgb.mode == "cached":
                from isl_slt.features.rgb import read_rgb
                read_rgb(config, record, dependencies, prepared.times, verify_assets=verify_assets, signatures=signatures)
        counts[name] = len(records)
    return {"verified_cache_counts": counts, "raw_assets_verified": verify_assets}


def prepare_model(config: Config, allow_network: bool) -> dict:
    if not allow_network:
        raise ValueError("Model preparation requires explicit --allow-network; ordinary operations stay local")
    results = {}
    if config.model.decoder == "t5":
        from huggingface_hub import HfApi, snapshot_download
        directory = Path(config.model.t5_dir)
        if (directory / "asset.json").exists():
            metadata = read_json(directory / "asset.json")
            if metadata["repository"] != config.model.t5_repo or metadata["requested_revision"] != config.model.t5_revision:
                raise ValueError("Existing prepared model uses a different repository/revision; choose a new asset directory")
            for name, expected in metadata["files"].items():
                if file_hash(directory / name) != expected:
                    raise ValueError(f"Prepared T5 asset checksum changed: {name}")
        else:
            revision = HfApi().model_info(config.model.t5_repo, revision=config.model.t5_revision).sha
            snapshot_download(repo_id=config.model.t5_repo, revision=revision, local_dir=str(directory),
                              allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer.*"])
            files = {str(path.relative_to(directory)): file_hash(path) for path in directory.rglob("*")
                     if path.is_file() and ".cache" not in path.relative_to(directory).parts and path.name != "asset.json"}
            metadata = {"repository": config.model.t5_repo, "requested_revision": config.model.t5_revision,
                        "resolved_revision": revision, "files": files,
                        "initializer": "generic pretrained text T5" if config.model.t5_repo == "google/flan-t5-small"
                        else "user-configured initializer; task-specific contamination provenance is unverified"}
            write_json(directory / "asset.json", metadata, immutable=True)
        results["t5"] = metadata
    if config.rgb.mode != "off":
        import urllib.request
        from isl_slt.utils.io import atomic_open, stream_identity
        path = Path(config.rgb.weights_path)
        if not path.exists():
            with urllib.request.urlopen("https://download.pytorch.org/models/resnet18-f37072fd.pth") as response:
                with atomic_open(path) as stream:
                    digest = stream_identity(response, destination=stream)["sha256"]
                    if not digest.startswith("f37072fd"):
                        raise ValueError("ResNet18 download hash-prefix check failed")
        else:
            digest = file_hash(path)
        if not digest.startswith("f37072fd"):
            raise ValueError("Local ResNet18 weights do not match the documented initializer")
        results["rgb"] = {"revision": config.rgb.revision, "sha256": digest}
    return results


def prepare_rgb(config: Config) -> dict:
    import torch
    from isl_slt.data.dataset import dependencies_for, raw_sequence
    from isl_slt.data.splits import checked_splits
    from isl_slt.features.rgb import RGBEncoder, load_crops, rgb_signature, write_rgb
    from isl_slt.utils.runtime import autocast, device_for
    if config.rgb.mode != "cached":
        raise ValueError("prepare-rgb writes frozen vectors and requires rgb.mode=cached")
    partitions, _ = checked_splits(config.data)
    device = device_for(config.runtime)
    encoder = RGBEncoder(config.rgb.weights_path).to(device).eval()
    counts = {}
    signatures = SignatureSession()
    for name, records in partitions.items():
        indexed = SourceIndex(records)
        for record in records:
            dependencies = dependencies_for(record, indexed, config)
            rgb_signature(config, record, dependencies, signatures=signatures)
            sequence = raw_sequence(record, indexed, config)
            images, masks = load_crops(record, indexed, sequence, config)
            with torch.no_grad(), autocast(config.runtime, device):
                vectors = encoder(images.to(device)).float().cpu().numpy()
            write_rgb(config, record, dependencies, vectors, masks.numpy(), sequence.times, signatures)
        counts[name] = len(records)
    return {"cached_examples": counts, "cnn_trainable": False, "revision": config.rgb.revision}


def evaluation_inputs(config: Config, checkpoint: str, partition: str):
    from isl_slt.data.dataset import PoseDataset
    from isl_slt.data.splits import checked_splits
    from isl_slt.features.cache import preprocessing_fingerprint
    from isl_slt.inference import Predictor
    partitions, metadata = checked_splits(config.data)
    predictor = Predictor.from_checkpoint(checkpoint, config.runtime)
    if preprocessing_fingerprint(config) != predictor.checkpoint["compatibility"]["feature_schema"]:
        raise ValueError("Evaluation feature/context configuration differs from checkpoint")
    if config.rgb.model_dump() != predictor.config.rgb.model_dump() or config.model.model_dump() != predictor.config.model.model_dump():
        raise ValueError("Use evaluation configuration inheriting the checkpoint experiment's model/RGB settings")
    saved_hash = predictor.checkpoint["compatibility"]["split_hashes"].get(partition)
    if saved_hash is None or saved_hash != metadata["hashes"][partition]:
        raise ValueError("Evaluation partition differs from the checkpoint's immutable manifest")
    dataset = PoseDataset(partitions[partition], config, training=False)
    statistics_signature = dataset.statistics["signature"] if dataset.statistics else None
    if statistics_signature != predictor.checkpoint["compatibility"].get("statistics"):
        raise ValueError("Evaluation feature statistics differ from the checkpoint")
    if config.rgb.mode == "cached" and dataset.rgb_initializer != predictor.checkpoint["rgb_initializer"]["sha256"]:
        raise ValueError("Evaluation RGB initializer differs from the checkpoint")
    return predictor, dataset, metadata


def comparison_fingerprint(predictor, effective_evaluation: dict) -> str:
    config = predictor.config
    training = config.training.model_dump(mode="json")
    training.pop("seed")
    return fingerprint({"model": config.model.model_dump(mode="json"), "training": training,
                        "features": predictor.checkpoint["compatibility"]["feature_schema"],
                        "rgb": config.rgb.model_dump(mode="json"), "lexical": config.lexical.model_dump(mode="json"),
                        "evaluation": effective_evaluation,
                        "rgb_initializer": predictor.checkpoint.get("rgb_initializer"),
                        "initializer": predictor.checkpoint["initializer"]})


def evaluate(config: Config, checkpoint: str, partition: str, output: str, decoding: str | None = None) -> dict:
    from isl_slt.evaluation.generation import generate_loader, score_rows, select_decoding
    from isl_slt.training.engine import loader_for
    predictor, dataset, metadata = evaluation_inputs(config, checkpoint, partition)
    loader, _ = loader_for(dataset, predictor.tokenizer, config, training=False)
    checkpoint_sha = file_hash(checkpoint)
    generation_hash = fingerprint(config.generation.model_dump(mode="json"))
    if partition == "validation":
        (beams, _, rows), candidates = select_decoding(predictor.model, predictor.tokenizer, loader, config, predictor.device)
        selection = {"checkpoint_sha256": checkpoint_sha, "partition": "validation", "selected_beams": beams,
                     "validation_membership_hash": metadata["hashes"]["validation"],
                     "generation_hash": generation_hash, "candidates": candidates}
        write_json(Path(output) / "decoding.json", selection)
    else:
        if decoding:
            selection = load_decoding_selection(decoding, checkpoint_sha, metadata["hashes"]["validation"], generation_hash)
            beams = selection["selected_beams"]
        else:
            if predictor.checkpoint["state"].get("decoding_policy") == "training-validation" and (
                set(predictor.config.generation.train_beams) != set(config.generation.beams)
            ):
                raise ValueError("Run the final validation beam sweep and supply --decoding before test/gold")
            if (config.generation.model_dump() != predictor.config.generation.model_dump() or predictor.beams is None
                or predictor.checkpoint["state"]["decoding_partition"] != "validation"):
                raise ValueError("Run validation decoding selection first and supply --decoding")
            beams = predictor.beams
        rows = generate_loader(predictor.model, predictor.tokenizer, loader, config, predictor.device, beams)
    write_jsonl(Path(output) / "predictions.jsonl", rows)
    metric_report = score_rows(rows, config)
    effective = {"generation": config.generation.model_dump(mode="json"), "selected_beams": beams,
                 "metrics": config.evaluation.metrics, "normalization": metric_report["normalization"],
                 "signatures": metric_report["signatures"],
                 "resource_identities": metric_report.get("resource_identities", {}),
                 "snapshot_precision": predictor.checkpoint.get("snapshot_precision", {"storage": "native"}),
                 "bertscore_model": config.evaluation.bertscore_model,
                 "bertscore_revision": config.evaluation.bertscore_revision,
                 "precision": config.runtime.precision, "recurrent_fp32": config.runtime.recurrent_fp32,
                 "deterministic": config.runtime.deterministic, "determinism_mode": config.runtime.determinism_mode}
    report = {"partition": partition, "membership_hash": metadata["hashes"][partition],
              "protocol": metadata["protocol"], "seed": predictor.config.training.seed,
              "comparison_fingerprint": comparison_fingerprint(predictor, effective), "selected_beams": beams,
              "effective_evaluation": effective, "evaluation_fingerprint": fingerprint(effective),
              "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": Path(checkpoint).stat().st_size,
              "checkpoint_kind": predictor.checkpoint.get("checkpoint_kind", "legacy"), "metrics": metric_report,
              "runtime": config.runtime.model_dump(mode="json"),
              "context_lookahead_seconds": config.context.next_seconds if config.context.enabled else 0,
              "parameter_counts": predictor.checkpoint["parameter_counts"]}
    write_json(Path(output) / "metrics.json", report)
    return report


def diagnostics(config: Config, checkpoint: str, partition: str, output: str, decoding: str | None = None) -> dict:
    from isl_slt.data.dataset import LengthBuckets
    from isl_slt.evaluation.diagnostics import perturb_items
    from isl_slt.evaluation.generation import score_rows
    predictor, dataset, metadata = evaluation_inputs(config, checkpoint, partition)
    predictor.configure_decoding(config.generation, decoding)
    if predictor.beams is None:
        raise ValueError("Diagnostics require validation-selected decoding from a trajectory checkpoint")
    import random
    order = list(range(len(dataset)))
    random.Random(config.evaluation.diagnostic_seed).shuffle(order)
    if len(order) < 2:
        raise ValueError("Paired diagnostics need at least two examples")
    donors = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    reports = {}
    from isl_slt.evaluation.metrics import MetricResources
    resources = MetricResources()
    for kind in ("real", "blank", "incorrect", "shuffle"):
        rows = []
        batches = LengthBuckets(dataset.lengths, config.training.micro_batch_size, config.training.max_frames_per_batch,
                                config.training.seed, False)
        for indices in batches:
            items, flags = [], []
            for index in indices:
                item = dataset[index]
                if kind == "incorrect":
                    donor = dataset[donors[index]]
                    flag = {"kind": kind, "donor_uid": donor["record"].uid,
                            "same_source": donor["record"].source_id == item["record"].source_id}
                    donor["record"] = item["record"]
                    changed = donor
                else:
                    changed_items, changed_flags = perturb_items([item], kind, config.evaluation.diagnostic_seed + index)
                    changed, flag = changed_items[0], changed_flags[0]
                items.append(changed)
                flags.append(flag)
            predictions = predictor.predict_items(items)
            for index, prediction, flag in zip(indices, predictions, flags):
                record = dataset.records[index]
                rows.append({**prediction, "source_id": record.source_id, "references": [record.text],
                             "clip_type": record.clip_type,
                             "interpreter_reference": record.interpreter_text, "diagnostic": flag})
        write_jsonl(Path(output) / f"{kind}.jsonl", rows)
        reports[kind] = score_rows(rows, config, resources=resources)
    result = {"paired_metrics": reports, "membership_hash": metadata["hashes"][partition],
              "seed": config.evaluation.diagnostic_seed,
              "interpretation": "Evidence of dependence on visual inputs; not standalone semantic-faithfulness proof"}
    write_json(Path(output) / "diagnostics.json", result)
    return result
