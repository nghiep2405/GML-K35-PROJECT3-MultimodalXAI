"""Integrity contracts for split, query pools, embeddings, negative candidates."""
from pathlib import Path
from collections import defaultdict
import pandas as pd
from .guard import WriteGuard, verify_frozen_val_test, verify_original_split, SPLIT_SHA256
from .sampler import EpochQuerySampler


def audit_pool(vg_root,exp_root,expected_train_images=83540,skip_split_checks=False):
    import duckdb
    guard=WriteGuard(vg_root,exp_root)
    val_checks = verify_frozen_val_test(vg_root,full_sha=True) if not skip_split_checks else {}
    split_checks = verify_original_split(vg_root) if not skip_split_checks else {}
    pool=Path(exp_root)/'data/query_pool/queries.parquet'
    original=Path(vg_root)/'clip_baseline/split_v2'
    con=duckdb.connect(database=':memory:')
    def qp(p):return str(Path(p).resolve()).replace("'", "''")
    q=con.execute(f"SELECT COUNT(*),COUNT(DISTINCT text_id),COUNT(DISTINCT positive_image_id) FROM read_parquet('{qp(pool)}')").fetchone()
    assert q[0]==q[1]
    if expected_train_images is not None:
        assert q[2]==expected_train_images, f'Expected {expected_train_images} train images, found {q[2]}'
    for sp in ('val','test'):
        other=original/sp/'queries.parquet'
        if other.is_file():
            matches=con.execute(f"SELECT COUNT(*) FROM read_parquet('{qp(pool)}') p JOIN read_parquet('{qp(other)}') v ON p.text_id=v.text_id").fetchone()[0]
            assert matches==0
    con.close()
    return {'status':'PASS','train_queries':q[0],'train_images':q[2],'frozen_val_test':val_checks,'original_split':split_checks}


def audit_sampling(exp_root,epochs=8,seed=2026,max_queries_per_epoch=None):
    pool=pd.read_parquet(Path(exp_root)/'data/query_pool/queries.parquet',columns=['positive_image_id','text_id'])
    sampler=EpochQuerySampler(pool.itertuples(index=False,name=None),seed=seed)
    sample_0=sampler.sample(0)
    assert sample_0==EpochQuerySampler(reversed(list(pool.itertuples(index=False,name=None))),seed=seed).sample(0)
    coverage=sampler.coverage(epochs)
    limit = max_queries_per_epoch if max_queries_per_epoch is not None else 83540 * 5
    assert all(x['queries']<=limit for x in coverage)
    return {'status':'PASS','first_epoch_queries':len(sample_0),'epochs':coverage}
