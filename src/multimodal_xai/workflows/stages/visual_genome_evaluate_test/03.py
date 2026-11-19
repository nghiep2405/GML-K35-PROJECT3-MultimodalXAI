from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

MODEL_ROOT=EXP_ROOT/'model/gat_dynamic_v2'
FEATURE_ROOT=CLIP_ROOT/'features'
CACHE_ROOT=VG_ROOT/'model/gat_v1/cache'       # READ ONLY: reuse previous compact cache
CACHE_PATH=CACHE_ROOT/'compact_graph_cache_v2.pt'
assert CACHE_PATH.is_file(), 'Original compact GAT graph cache missing. STOP; never rebuild in old tree.'
compact=torch.load(CACHE_PATH,map_location='cpu',weights_only=False)
assert compact['node_ptr'][-1].item()==2_609_364
assert compact['edge_ptr'][-1].item()==8_005_044
print('Loaded original compact graph cache read-only:',CACHE_PATH)
