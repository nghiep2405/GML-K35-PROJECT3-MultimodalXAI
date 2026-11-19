"""Compare Flickr30k replay output with the original locked summary."""
import csv
import json
import math


def compare_flickr_summaries(reference, actual, output, tolerance=2e-4):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    old = json.loads(reference.read_text(encoding="utf-8"))
    new = json.loads(actual.read_text(encoding="utf-8"))
    old_runs = {run["run"]: run for run in old["runs"]}
    new_runs = {run["run"]: run for run in new["runs"]}
    rows = []

    def record(run, metric, expected, observed, numeric=False):
        delta = abs(expected - observed) if numeric and observed is not None else None
        ok = delta <= tolerance if delta is not None else expected == observed
        rows.append((run, metric, expected, observed, delta, tolerance if numeric else 0,
                     "PASS" if ok else "FAIL"))

    for field in ("split", "candidate_images", "test_queries", "score_formula"):
        record("protocol", field, old[field], new.get(field))
    for name in sorted(set(old_runs) | set(new_runs)):
        record(name, "present", name in old_runs, name in new_runs)
        if name not in old_runs or name not in new_runs:
            continue
        expected, observed = old_runs[name], new_runs[name]
        for key in ("seed", "edge_mode", "checkpoint_epoch", "alpha_graph_locked_from_validation"):
            record(name, key, expected[key], observed.get(key))
        for section in ("test_clip_only", "test_clip_plus_graph"):
            for metric, value in expected[section].items():
                record(name, f"{section}.{metric}", value, observed.get(section, {}).get(metric), True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("run", "metric", "expected", "actual", "absolute_delta", "tolerance", "status"))
        writer.writerows(rows)
    return all(row[-1] == "PASS" for row in rows)
