import os, sys, subprocess, importlib

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
