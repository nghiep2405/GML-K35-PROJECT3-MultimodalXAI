import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "gdrive-fsspec", "fsspec", "pyarrow", "duckdb",
    "torch-geometric", "pandas", "tqdm",
])
vg_deps_ready = True
