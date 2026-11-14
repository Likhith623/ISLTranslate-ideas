"""Build a reproducible source archive excluding data, weights, caches and environments."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="releases/PCAT-SLT.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "data", "runs",
                "evaluation", "build", "dist", "releases"}
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and path != output and not path.is_symlink()
                and not any(part in excluded or part.endswith(".egg-info") for part in relative.parts)
                and path.suffix not in {".pyc", ".pt", ".npz", ".zip"} and path.name != ".DS_Store"):
            files.append((relative, path.read_bytes()))
    checksums = {str(relative): hashlib.sha256(content).hexdigest() for relative, content in files}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, content in files:
            info = zipfile.ZipInfo(f"{root.name}/{relative.as_posix()}", date_time=(2026, 11, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
        info = zipfile.ZipInfo(f"{root.name}/MANIFEST.sha256.json", date_time=(2026, 11, 14, 0, 0, 0))
        archive.writestr(info, json.dumps(checksums, indent=2).encode())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "source_files": len(files), "bytes": output.stat().st_size,
                      "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
