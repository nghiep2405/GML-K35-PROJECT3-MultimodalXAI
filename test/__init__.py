"""Allow the fixture suite to run via unittest without optional test packages."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
