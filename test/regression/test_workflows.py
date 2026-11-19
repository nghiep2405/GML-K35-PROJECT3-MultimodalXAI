import ast
from importlib.resources import files
import json
from pathlib import Path
import unittest

from multimodal_xai.workflows import runner


class WorkflowTests(unittest.TestCase):
    def test_stages_compile_and_notebooks_reference_catalog(self):
        notebooks = Path(__file__).resolve().parents[2] / "src/notebooks"
        wrappers = {path.stem: path for path in notebooks.rglob("*.ipynb")}
        self.assertEqual(set(wrappers), set(runner.catalog()))
        resources = files(runner.__package__).joinpath("stages")
        for name, entry in runner.catalog().items():
            with self.subTest(name=name):
                nb = json.loads(wrappers[name].read_text(encoding="utf-8"))
                cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                self.assertEqual(len(cells), len(entry["stages"]))
                for cell, stage in zip(cells, entry["stages"]):
                    self.assertIn(f"run_stage({name!r}, {stage['cell']}, globals())", "".join(cell["source"]))
                    resource = resources.joinpath(stage.get("location", name), stage["file"])
                    self.assertTrue(resource.is_file())
                    if stage["file"].endswith(".py"):
                        ast.parse(resource.read_text(encoding="utf-8"), filename=str(resource))
        for resource in resources.joinpath("shared_kg_xai").iterdir():
            ast.parse(resource.read_text(encoding="utf-8"), filename=str(resource))

    def test_runtime_executes_shared_functions_in_caller_namespace(self):
        namespace = {}
        runner.run_shared("artifacts", namespace)
        self.assertIn("latest_locked", namespace)
        self.assertIs(namespace["latest_locked"].__globals__, namespace)
