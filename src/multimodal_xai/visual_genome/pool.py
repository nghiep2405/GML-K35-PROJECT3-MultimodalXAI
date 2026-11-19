"""Create train-only 20-query pool from immutable processed data using DuckDB."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .guard import WriteGuard, SPLIT_SHA256, sha256


def sqlpath(path):return str(Path(path).resolve()).replace("'", "''")


def build_query_pool(vg_root: Path, exp_root: Path, per_image=20, seed=2026,
                     expected_train_images: int | None = 83540,
                     expected_split_sha: str | None = SPLIT_SHA256):
    import duckdb
    if per_image != 20: raise ValueError('v2 contract: max pool 20 queries/image')
    guard=WriteGuard(vg_root,exp_root)
    original=Path(vg_root)/'processed'
    for path in ['splits_v2.parquet','texts.parquet','image_text_edges.parquet']:
        guard.assert_original(original/path)
        if not (original/path).is_file(): raise FileNotFoundError(original/path)
    target=guard.mkdir(Path(exp_root)/'data/query_pool')
    split_sha = expected_split_sha or SPLIT_SHA256
    # Safe rerun: reuse only completed artifacts whose byte hashes match v2 manifest.
    old_manifest=target/'manifest.json'
    if old_manifest.exists():
        report=json.loads(old_manifest.read_text())
        assert report['split_sha256']==split_sha and report['pool_cap']==per_image and report['seed']==seed
        for name,h in report['artifacts'].items():
            assert sha256(target/name)==h, f'Existing v2 pool mutated: {name}'
        return report
    if (target/'queries.parquet').exists() or (target/'ground_truth.parquet').exists():
        raise RuntimeError('Incomplete v2 pool present; review incomplete files manually before retrying')
    import pyarrow.parquet as pq
    fields=set(pq.read_schema(original/'splits_v2.parquet').names)
    image_id_col='item_id' if 'item_id' in fields else 'image_id'
    where="WHERE item_type='image'" if 'item_type' in fields else ''
    text_fields=set(pq.read_schema(original/'texts.parquet').names)
    query_expr=("COALESCE(NULLIF(trim(CAST(t.text_raw AS VARCHAR)),''), NULLIF(trim(CAST(t.text_norm AS VARCHAR)),''))"
                if 'text_raw' in text_fields else "NULLIF(trim(CAST(t.text_norm AS VARCHAR)),'')")
    s=sqlpath(original/'splits_v2.parquet');t=sqlpath(original/'texts.parquet');e=sqlpath(original/'image_text_edges.parquet')
    connection=duckdb.connect(database=':memory:')
    split=f"SELECT CAST({image_id_col} AS VARCHAR) image_id, CAST(split AS VARCHAR) split FROM read_parquet('{s}') {where}"
    gts=f"""SELECT DISTINCT CAST(e.text_id AS VARCHAR) text_id, CAST(e.image_id AS VARCHAR) image_id
           FROM read_parquet('{e}') e JOIN ({split}) sp
           ON CAST(e.image_id AS VARCHAR)=sp.image_id WHERE sp.split='train'"""
    originals=Path(vg_root)/'clip_baseline/split_v2/train_hard_negatives/queries.parquet'
    orig_join = f"LEFT JOIN (SELECT CAST(text_id AS VARCHAR) text_id FROM read_parquet('{sqlpath(originals)}')) orig ON valid.text_id=orig.text_id" if originals.is_file() else ""
    orig_order = "(CASE WHEN orig.text_id IS NOT NULL THEN 0 ELSE 1 END), " if originals.is_file() else ""
    # Match original CLIP protocol's MIN(image_id) canonical positive rule, with existing train queries prioritized.
    pool_sql=f"""WITH gt AS ({gts}), valid AS (
         SELECT CAST(t.text_id AS VARCHAR) text_id, {query_expr} query_text,
           CAST(t.text_type AS VARCHAR) text_type, MIN(gt.image_id) positive_image_id
         FROM read_parquet('{t}') t JOIN gt ON CAST(t.text_id AS VARCHAR)=gt.text_id
         WHERE {query_expr} IS NOT NULL
         GROUP BY CAST(t.text_id AS VARCHAR), {query_expr}, CAST(t.text_type AS VARCHAR)
       ), ranked AS (
         SELECT valid.*, ROW_NUMBER() OVER (PARTITION BY valid.positive_image_id
                   ORDER BY {orig_order}md5(valid.text_id || ':{seed}'), valid.text_id) rn
         FROM valid {orig_join}
       ) SELECT text_id, query_text, text_type, positive_image_id
         FROM ranked WHERE rn<={per_image} ORDER BY text_id, positive_image_id"""
    def copy(sql,path):
        dest=guard.writable(path)
        if dest.exists():raise FileExistsError(f'Existing output requires explicit new experiment name: {dest}')
        tmp=guard.writable(dest.with_name(dest.name+'.tmp'))
        if tmp.exists(): tmp.unlink()
        connection.execute(f"COPY ({sql}) TO '{sqlpath(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        tmp.replace(dest)
    copy(pool_sql,target/'queries.parquet')
    gt_sql=f"""WITH gt AS ({gts}) SELECT gt.text_id, gt.image_id
       FROM gt JOIN read_parquet('{sqlpath(target/'queries.parquet')}') p
       ON gt.text_id=p.text_id ORDER BY gt.text_id, gt.image_id"""
    copy(gt_sql,target/'ground_truth.parquet')
    if originals.is_file():
        overlap=connection.execute(f"""SELECT COUNT(*) FROM
            (SELECT CAST(text_id AS VARCHAR) text_id FROM read_parquet('{sqlpath(originals)}')
             EXCEPT SELECT text_id FROM read_parquet('{sqlpath(target/'queries.parquet')}'))""").fetchone()[0]
        assert overlap==0, f'Previously-used original train queries lost: {overlap}'
    counts=connection.execute(f"SELECT COUNT(*) n FROM read_parquet('{sqlpath(target/'queries.parquet')}') GROUP BY positive_image_id").fetchdf().n
    n_images=connection.execute(f"SELECT COUNT(*) FROM ({split}) WHERE split='train'").fetchone()[0]
    train_queries=connection.execute(f"SELECT COUNT(*) FROM read_parquet('{sqlpath(target/'queries.parquet')}')").fetchone()[0]
    if expected_train_images is not None:
        assert n_images==expected_train_images, f'Expected {expected_train_images} train images, found {n_images}'
    assert counts.size==n_images and counts.min()>=1 and counts.max()<=20
    # Validate all text IDs against source eval protocol, and all positive IDs train.
    for split_name in ['val','test']:
        old=Path(vg_root)/f'clip_baseline/split_v2/{split_name}/queries.parquet'
        if old.is_file():
            collisions=connection.execute(f"""SELECT COUNT(*) FROM read_parquet('{sqlpath(target/'queries.parquet')}') p
                    JOIN read_parquet('{sqlpath(old)}') q ON p.text_id=CAST(q.text_id AS VARCHAR)""").fetchone()[0]
            assert collisions==0, f'Query leakage to {split_name}: {collisions}'
    wrong=connection.execute(f"""SELECT COUNT(*) FROM read_parquet('{sqlpath(target/'queries.parquet')}') q
          LEFT JOIN ({split}) sp ON q.positive_image_id=sp.image_id WHERE sp.split IS DISTINCT FROM 'train'""").fetchone()[0]
    assert wrong==0
    report={'split_sha256':split_sha,'seed':seed,'pool_cap':per_image,'images':int(n_images),'queries':int(train_queries),
      'queries_per_image':{'min':int(counts.min()),'median':float(counts.median()),'mean':float(counts.mean()),'max':int(counts.max())},
      'original_5_query_subset':True,'train_only':True,'text_id_overlap_val_test':0,
      'artifacts':{p.name:sha256(p) for p in [target/'queries.parquet',target/'ground_truth.parquet']}}
    guard.save_json(target/'manifest.json',report)
    connection.close()
    return report
