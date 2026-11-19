"""Reuses original query embeddings by ID; encodes ONLY new text IDs; sharded resume."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .guard import WriteGuard, sha256


def prepare_text_embeddings(vg_root, exp_root, shard_rows=16384, batch_size=512, device=None, encoder_fn=None):
    import torch
    import torch.nn.functional as F
    src=Path(vg_root)/'clip_baseline/split_v2/train_hard_negatives'
    out=Path(exp_root)/'data/embeddings'
    guard=WriteGuard(vg_root,exp_root);guard.mkdir(out/'shards');guard.mkdir(out/'indexes')
    queries=pd.read_parquet(Path(exp_root)/'data/query_pool/queries.parquet')
    mapping = {}
    old_embeddings = None
    if (src/'text_embedding_index.parquet').is_file() and (src/'text_embeddings.pt').is_file():
        original_index=pd.read_parquet(src/'text_embedding_index.parquet')
        original_index['text_id']=original_index.text_id.astype(str)
        mapping=dict(zip(original_index.text_id,original_index.embedding_row))
        if len(mapping)!=len(original_index):raise ValueError('Original text embedding IDs duplicate')
        assert sorted(map(int,original_index.embedding_row))==list(range(len(original_index))), 'Noncontiguous original embedding rows'
        old_embeddings=torch.load(src/'text_embeddings.pt',map_location='cpu',weights_only=True)
        assert old_embeddings.ndim==2 and old_embeddings.shape[0]==len(original_index) and old_embeddings.shape[1]==512
    if device is None:device='cuda' if torch.cuda.is_available() else 'cpu'
    model=None;tokenizer=None;written=0;reused=0;computed=0
    expected_files=[]
    for shard_idx,start in enumerate(range(0,len(queries),shard_rows)):
        part=queries.iloc[start:start+shard_rows]
        target=guard.writable(out/'shards'/f'part_{shard_idx:05d}.pt')
        index=guard.writable(out/'indexes'/f'part_{shard_idx:05d}.parquet')
        expected_ids=part.text_id.astype(str).tolist()
        if target.exists() and index.exists():
            try:
                idx=pd.read_parquet(index)
                if idx.text_id.astype(str).tolist()==expected_ids and idx.embedding_row.tolist()==list(range(len(part))):
                    z=torch.load(target,map_location='cpu',weights_only=True)
                    if z.shape==(len(part),512) and torch.isfinite(z).all():
                        expected_files.extend([target,index]);continue
            except Exception:
                pass
        # Clean up any orphaned incomplete shard from an interrupted write
        if target.exists(): target.unlink(missing_ok=True)
        if index.exists(): index.unlink(missing_ok=True)
        arr=torch.empty((len(part),512),dtype=torch.float16)
        todo=[];ids=part.text_id.astype(str).tolist()
        for j,tid in enumerate(ids):
            if tid in mapping and old_embeddings is not None:
                arr[j]=old_embeddings[mapping[tid]];reused+=1
            else:todo.append(j)
        if todo:
            if encoder_fn is not None:
                for b in range(0,len(todo),batch_size):
                    loc=todo[b:b+batch_size]
                    text=part.iloc[loc].query_text.astype(str).tolist()
                    z=encoder_fn(text)
                    arr[loc]=z.cpu().half();computed+=len(loc)
            else:
                import open_clip
                if model is None:
                    model,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device=device)
                    tokenizer=open_clip.get_tokenizer('ViT-B-32');model.eval()
                with torch.inference_mode():
                    for b in range(0,len(todo),batch_size):
                        loc=todo[b:b+batch_size]
                        text=part.iloc[loc].query_text.astype(str).tolist()
                        tokens=tokenizer(text).to(device)
                        z=F.normalize(model.encode_text(tokens).float(),dim=1)
                        arr[loc]=z.cpu().half();computed+=len(loc)
        assert torch.isfinite(arr).all()
        tmp=guard.writable(target.with_name(target.name+'.tmp'))
        torch.save(arr,tmp);tmp.replace(target)
        idx=pd.DataFrame({'text_id':ids,'embedding_row':np.arange(len(ids),dtype=np.int64),'shard':shard_idx})
        tm=guard.writable(index.with_name(index.name+'.tmp'))
        idx.to_parquet(tm,index=False);tm.replace(index)
        expected_files.extend([target,index]);written+=1
        print(f'embeddings shard {shard_idx}: {len(part)} ids | reused={reused} new={computed}')
    report={'model':'ViT-B-32','pretrained':'openai','feature_dim':512,'dtype':'float16',
            'queries':len(queries),'shards':len(expected_files)//2,'new_shards':written,
            'reused_this_invocation':reused,'computed_this_invocation':computed,
            'completed':True,
            'parts':[{'file':p.name,'sha256':sha256(p)} for p in expected_files if p.suffix=='.pt']}
    guard.save_json(out/'manifest.json',report)
    return report
