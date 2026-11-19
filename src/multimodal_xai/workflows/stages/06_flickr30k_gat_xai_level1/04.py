# Dùng NGUYÊN mã của nhóm (run_gat.py, prepare_retrieval.py) để giải thích đúng cơ chế đã huấn luyện.
# Mã được chép vào thư mục làm việc để import, tránh ghi __pycache__ hay thư mục run vào Drive.
import shutil, hashlib

os.environ["GRAPHGAT_PROJECT_ROOT"] = str(FINAL)
os.environ["FLICKR_EDGE_MODE"] = "real_edges"

import json, math, random, time
from IPython.display import HTML, display
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from multimodal_xai.flickr30k import train as gat
from multimodal_xai.flickr30k import data as flickr_data
importlib.reload(flickr_data)
importlib.reload(gat)
DEVICE = gat.DEVICE
torch.set_grad_enabled(False)
print("torch", torch.__version__, "| device:", DEVICE)

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
