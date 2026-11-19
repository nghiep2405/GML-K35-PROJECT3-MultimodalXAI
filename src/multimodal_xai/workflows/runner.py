"""Run extracted stages with the state of the calling notebook or CLI session."""
from functools import lru_cache
from importlib.resources import files
import json


@lru_cache
def catalog():
    resource = files(__package__).joinpath("catalog.json")
    entries = json.loads(resource.read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in entries}


def _execute(resource, namespace):
    source = resource.read_text(encoding="utf-8")
    if resource.name.endswith(".ipy"):
        from IPython import get_ipython
        from IPython.core.interactiveshell import InteractiveShell
        shell = get_ipython() or InteractiveShell.instance()
        source = shell.transform_cell(source)
        namespace.setdefault("get_ipython", lambda: shell)
    namespace.setdefault("__name__", "__main__")
    exec(compile(source, str(resource), "exec"), namespace)


def run_stage(workflow, cell, namespace):
    entry = catalog()[workflow]
    stage = next((item for item in entry["stages"] if item["cell"] == cell), None)
    if stage is None:
        raise ValueError(f"{workflow} has no code stage {cell}")
    path = files(__package__).joinpath("stages", stage.get("location", workflow), stage["file"])
    _execute(path, namespace)


def run_shared(group, namespace):
    if group not in {"artifacts", "models", "ranking", "explain"}:
        raise ValueError(f"Unknown shared XAI group: {group}")
    path = files(__package__).joinpath("stages", "shared_kg_xai", group + ".py")
    _execute(path, namespace)


def run_workflow(workflow, through=None):
    stages = catalog()[workflow]["stages"]
    if through is not None and through not in {stage["cell"] for stage in stages}:
        raise ValueError(f"{workflow} has no code stage {through}")
    namespace = {}
    for stage in stages:
        cell = stage["cell"]
        print(f"[{workflow}:{cell:02d}] {stage['label']}", flush=True)
        run_stage(workflow, cell, namespace)
        if cell == through:
            break
    return namespace
