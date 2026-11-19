from pathlib import Path
import os,sys,json,time,hashlib,random,subprocess,math,shutil
import torch
assert torch.cuda.is_available(), 'Chọn Runtime > Change runtime type > GPU'
subprocess.run([sys.executable,'-m','pip','install','-q','pandas','pyarrow','transformers','huggingface_hub','matplotlib','tqdm'],check=True)
import numpy as np,pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import display
if not os.environ.get('ADAMF_DATA_ROOT'):
    from google.colab import drive
    drive.mount('/content/drive')
ROOT=Path(os.environ.get('ADAMF_DATA_ROOT', '/content/drive/MyDrive/MLGs - Project/dataset/adamf')).resolve()
CLEAN=ROOT/'processed/db15k/experimental_clean_v1'
RETRIEVAL=ROOT/'db15k_image_retrieval/runs/db15k_clean_v1'
ABLATION='random_graph'
RUN=ROOT/'model/adamf_v1/db15k_clean_kgc_to_retrieval_seed2027_ablation_random_graph_v1'
if os.environ.get('ADAMF_RUN_ROOT'):
    RUN=Path(os.environ['ADAMF_RUN_ROOT']).expanduser().resolve() / RUN.name
RUN.mkdir(parents=True,exist_ok=True)
CFG=dict(epochs=500,seed=2027,dim=250,margin=12.,lr=1e-4,batch_size=256,
    negative_count=128,adv_temperature=2.,mu=0.,freeze_features=True,
    topk=1000,alpha_grid=[i/10 for i in range(11)],
    checkpoint_batches=50,session_hours=8.,visual_policy='train_image_CLIP_mean_L2',
    direct_policy='CLIP_only',selection='combined_validation_retrieval_MRR')
DEVICE=torch.device('cuda')
random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
torch.backends.cuda.matmul.allow_tf32=False
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(obj,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
def save_tensor(obj,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def resolve(value):
    p=(ROOT/value).resolve();assert p.is_relative_to(ROOT),p;assert p.is_file(),p;return p
HASHES={}
def verified(key,spec):
    p=resolve(spec.get('phase2_relative_path',spec.get('path')))
    actual=sha(p);assert actual==spec['sha256'],f'Hash mismatch: {key}'
    HASHES[key]=actual;print(key,'->',p);return p
print('DATA:',ROOT,'\nRUN:',RUN,'\nGPU:',torch.cuda.get_device_name(0))
print(json.dumps(CFG,indent=2))
CFG['ablation']=ABLATION

CFG.pop('checkpoint_batches', None)
CFG['checkpoint_every_epochs']=30


SEED_RUN=RUN/f"seed_{CFG['seed']}"
CACHE_DIR=SEED_RUN/'cache'
GRAPH_DIR=SEED_RUN/'graph'
VALIDATION_DIR=SEED_RUN/'validation'
RESULTS_DIR=SEED_RUN/'results'
CHECKPOINTS=SEED_RUN/'checkpoints'
for folder in [SEED_RUN,CACHE_DIR,GRAPH_DIR,VALIDATION_DIR,RESULTS_DIR,CHECKPOINTS]:
    folder.mkdir(parents=True,exist_ok=True)
