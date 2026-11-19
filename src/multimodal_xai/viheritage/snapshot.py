"""Read-only verification of a published ViHeritage-MMKG snapshot."""
from pathlib import Path
import json

from ..visual_genome.guard import sha256


def verify_snapshot(dataset: Path) -> dict:
    dataset = dataset.resolve()
    split_path = dataset / "processed/split_manifest.json"
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"] if "artifacts" in manifest else manifest["files"]
    issues = []
    for key, artifact in artifacts.items():
        file = (dataset / artifact["path"]).resolve()
        if not file.is_relative_to(dataset):
            issues.append({"name": key, "error": "outside dataset"})
        elif not file.is_file():
            issues.append({"name": key, "error": "missing"})
        elif sha256(file) != artifact["sha256"]:
            issues.append({"name": key, "error": "sha256 mismatch"})
    return {"status": "FAIL" if issues else "PASS", "checked": len(artifacts),
            "counts": manifest["counts"], "issues": issues}
