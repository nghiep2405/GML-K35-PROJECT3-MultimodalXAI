"""Mine full train-only CLIP hard negatives for all v2 queries using per-shard tensors."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .guard import WriteGuard, sha256


def mine_one_batch(query_z, image_z, image_ids, positive_positions, k=50):
    """CPU/GPU torch tensors. Compute exact positive rank on full train candidate set."""
    import torch
    if k<1:raise ValueError('k must be >=1')
    logits=query_z.float() @ image_z.float().T
    rows=[]
    for j,indices in enumerate(positive_positions):
        if not indices:raise ValueError('Query without known positives')
        unique=sorted(set(int(x) for x in indices))
        raw=logits[j]
        best=float(raw[unique].max().item())
        positive_rank=1+int((raw>best).sum().item())
        assert len(image_ids)-len(unique)>=k
        negatives=raw.clone()
        negatives[unique]=-float('inf')
        scores,inds=torch.topk(negatives,k=k,sorted=True)
        neg=[image_ids[int(x)] for x in inds.tolist()]
        assert len(neg)==len(set(neg)) and all(x not in {image_ids[p] for p in unique} for x in neg)
        rows.append({'positive_score':best,'positive_rank':positive_rank,
                     'negative_image_ids':neg,'negative_scores':[float(x) for x in scores.tolist()]})
    return rows


def build_hard_negatives(vg_root,exp_root,batch_size=128,device=None,k=50,expected_train_images=83540):
    import torch
    import pyarrow.parquet as pq
    original=Path(vg_root)/'clip_baseline/split_v2/train_hard_negatives'
    work=Path(exp_root)/'data'
    guard=WriteGuard(vg_root,exp_root);out=guard.mkdir(work/'hard_negatives/parts')
    pool=pd.read_parquet(work/'query_pool/queries.parquet')
    gt=pd.read_parquet(work/'query_pool/ground_truth.parquet')
    known=defaultdict(set)
    for r in gt[['text_id','image_id']].itertuples(index=False):known[str(r.text_id)].add(str(r.image_id))
    train_img=pd.read_parquet(original/'image_embedding_index.parquet')
    train_ids=train_img.image_id.astype(str).tolist()
    if expected_train_images is not None:
        assert len(train_ids)==expected_train_images, f'Expected {expected_train_images} train images, found {len(train_ids)}'
    assert len(set(train_ids))==len(train_ids)
    positions={image_id:i for i,image_id in enumerate(train_ids)}
    image_z=torch.load(original/'image_embeddings.pt',map_location='cpu',weights_only=True)
    assert image_z.shape==(len(train_ids),512)
    if device is None:device='cuda' if torch.cuda.is_available() else 'cpu'
    image_z=image_z.to(device).float()
    expected=[];generated=0
    idx_parts=sorted((work/'embeddings/indexes').glob('part_*.parquet'))
    assert idx_parts, 'Missing embedding shards; run preparation first'
    for idx_path in idx_parts:
        stem=idx_path.stem
        target=guard.writable(out/f'{stem}.parquet')
        ix=pd.read_parquet(idx_path)
        # Resolve source rows using text ID, not inferred ordinal ranges.
        part=ix.merge(pool,on='text_id',how='left',validate='one_to_one',sort=False)
        assert len(part)==len(ix) and part.positive_image_id.notna().all()
        if target.exists():
            try:
                existing=pd.read_parquet(target,columns=['text_id'])
                if existing.text_id.astype(str).tolist()==part.text_id.astype(str).tolist():
                    expected.append(target);continue
            except Exception:
                pass
        target.unlink(missing_ok=True)
        tz=torch.load(work/'embeddings/shards'/f'{stem}.pt',map_location='cpu',weights_only=True)
        assert tz.shape==(len(part),512)
        records=[]
        for start in range(0,len(part),batch_size):
            batch=part.iloc[start:start+batch_size]
            z=tz[start:start+len(batch)].to(device)
            positives=[[positions[x] for x in known[str(tid)]] for tid in batch.text_id]
            results=mine_one_batch(z,image_z,train_ids,positives,k=k)
            for row,result in zip(batch.itertuples(index=False),results):
                records.append({'text_id':str(row.text_id),'positive_image_id':str(row.positive_image_id),**result})
        df=pd.DataFrame(records)
        assert len(df)==len(part)
        tmp=guard.writable(target.with_name(target.name+'.tmp'))
        df.to_parquet(tmp,index=False);tmp.replace(target)
        expected.append(target);generated+=1
        print(f'negative shard {stem}: {len(df)} rows')
    # Full integrity validation, no accidental reuse of corrupted old shards.
    import duckdb
    con=duckdb.connect(database=':memory:')
    escaped_parts = str(out/'part_*.parquet').replace("'", "''")
    counts=con.execute(f"SELECT count(*),count(DISTINCT text_id) FROM read_parquet('{escaped_parts}')").fetchone()
    assert counts==(len(pool),len(pool)),f'Hard-negative pool coverage mismatch: {counts}'
    # Audit via streaming batches (nested arrays), never trust presence of shard only.
    seen=set();total=0
    for p in expected:
        table=pq.ParquetFile(p)
        for batch in table.iter_batches(batch_size=4096):
            for r in batch.to_pylist():
                tid=str(r['text_id']);pimg=str(r['positive_image_id'])
                assert tid not in seen and pimg in known[tid]
                seen.add(tid);neg=list(map(str,r['negative_image_ids']))
                assert len(neg)==k and len(neg)==len(set(neg))
                assert set(neg).isdisjoint(known[tid]) and set(neg).issubset(positions)
                assert len(neg)==len(r['negative_scores'])
                total+=1
    assert total==len(pool)
    report={'queries':int(total),'negative_count':k,'unique_train_only':True,
            'all_known_positives_excluded':True,'embedding_alignment':'validated_by_matching_shard_indices',
            'parts':[{'name':p.name,'sha256':sha256(p)} for p in expected], 'generated_this_invocation':generated}
    guard.save_json(work/'hard_negatives/manifest.json',report)
    con.close()
    return report
