#!/usr/bin/env python3
"""Train the Visual Genome CLIP + edge-aware GAT reranker.

This is the canonical source used by the submitted main and ablation notebooks.
The frozen CLIP/graph artifacts are read from ``dataset/visual_genome``;
checkpoints and histories are written only below ``experiment_v2``.
Configuration is supplied through the ``GRAPHGAT_*`` environment variables
documented in README.md.
"""


def main():
    from pathlib import Path
    import os
    import sys
    import json
    import shutil

    PROJECT_ROOT = Path(os.environ.get('GRAPHGAT_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
    VG_ROOT = PROJECT_ROOT / 'dataset/visual_genome'
    EXP_ROOT = PROJECT_ROOT / 'experiment_v2'
    if not VG_ROOT.is_dir() or not EXP_ROOT.is_dir():
        raise FileNotFoundError(f'Expected {VG_ROOT} and {EXP_ROOT}')
    from .guard import WriteGuard, verify_frozen_val_test, SPLIT_SHA256
    from .model import EdgeAwareGAT
    OUTPUT_ROOT = Path(os.environ.get('GRAPHGAT_OUTPUT_ROOT', EXP_ROOT)).resolve()
    GUARD=WriteGuard(VG_ROOT,OUTPUT_ROOT)
    GRAPH_ROOT = VG_ROOT / 'graph/split_v2'
    CLIP_ROOT = VG_ROOT / 'clip_baseline/split_v2'
    PROCESSED_ROOT = VG_ROOT / 'processed'
    print('Immutable input:', VG_ROOT)
    print('Isolated output:', OUTPUT_ROOT)

    # ============================================
    # Runtime logging utilities
    # ============================================

    import time
    from datetime import datetime
    from tqdm.auto import tqdm

    def log_status(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        tqdm.write(f"[{timestamp}] {message}")

    log_status("GML Experiment v2 initialized")
    log_status(f"Input (read-only): {VG_ROOT}")
    log_status(f"Output (isolated): {OUTPUT_ROOT}")






    from pathlib import Path
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch

    MODEL_ROOT=OUTPUT_ROOT/'model/gat_dynamic_v2'
    FEATURE_ROOT=CLIP_ROOT/'features'
    CACHE_ROOT=VG_ROOT/'model/gat_v1/cache'       # READ ONLY: reuse previous compact cache
    CACHE_PATH=CACHE_ROOT/'compact_graph_cache_v2.pt'
    assert CACHE_PATH.is_file(), 'Original compact GAT graph cache missing. STOP; never rebuild in old tree.'
    compact=torch.load(CACHE_PATH,map_location='cpu',weights_only=False)
    assert compact['node_ptr'][-1].item()==2_609_364
    assert compact['edge_ptr'][-1].item()==8_005_044
    print('Loaded original compact graph cache read-only:',CACHE_PATH)




    import random
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from tqdm.auto import tqdm

    SEED = 2026
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)

    image_features = torch.load(
        FEATURE_ROOT / 'image_features.pt', map_location='cpu', weights_only=True
    ).float()
    entity_features = torch.load(
        FEATURE_ROOT / 'entity_label_features.pt', map_location='cpu', weights_only=True
    ).float()
    relation_features = torch.load(
        FEATURE_ROOT / 'relation_text_features.pt', map_location='cpu', weights_only=True
    ).float()
    relation_mapping = pd.read_parquet(GRAPH_ROOT / 'relation_mapping.parquet')
    relation_feature_index = pd.read_parquet(
        FEATURE_ROOT / 'relation_feature_index.parquet'
    )
    relation_join = relation_mapping.merge(
        relation_feature_index[['edge_family', 'relation', 'feature_row']],
        on=['edge_family', 'relation'], how='left', validate='one_to_one'
    )
    relation_type_features = torch.empty(
        (int(relation_mapping.relation_index.max()) + 1, 512), dtype=torch.float32
    )
    known = relation_join.feature_row.notna().to_numpy()
    known_type = torch.from_numpy(
        relation_join.loc[known, 'relation_index'].to_numpy(np.int64)
    )
    known_row = torch.from_numpy(
        relation_join.loc[known, 'feature_row'].to_numpy(np.int64)
    )
    relation_type_features[known_type] = relation_features[known_row]
    unk_feature_rows = relation_feature_index.loc[
        relation_feature_index.learnable_relation_index == 1, 'feature_row'
    ].to_numpy(np.int64)
    unk_feature = F.normalize(
        relation_features[torch.from_numpy(unk_feature_rows)].mean(0), dim=0
    )
    missing_types = relation_join.loc[~known, 'relation_index'].to_numpy(np.int64)
    assert set(missing_types).issubset({1})
    if len(missing_types):
        relation_type_features[torch.from_numpy(missing_types)] = unk_feature
    assert torch.isfinite(relation_type_features).all()

    image_index = pd.read_parquet(
        FEATURE_ROOT / 'image_feature_index.parquet', columns=['image_id', 'feature_row']
    )
    image_to_component = dict(zip(image_index.image_id, image_index.feature_row))
    global_to_local = torch.full((2_609_364,), -1, dtype=torch.long)
    print('Feature tensors loaded')




    def build_component_batch(component_rows):
        component_rows = list(dict.fromkeys(int(x) for x in component_rows))
        node_chunks, edge_chunks, relation_chunks, reverse_chunks = [], [], [], []
        batch_chunks, root_globals = [], []

        for batch_id, component in enumerate(component_rows):
            ns, ne = compact['node_ptr'][component:component + 2].tolist()
            es, ee = compact['edge_ptr'][component:component + 2].tolist()
            nodes = compact['node_perm'][ns:ne]
            edges = compact['edge_index'][:, es:ee]
            if os.environ.get('GRAPHGAT_EDGE_MODE', 'real_edges').strip().lower() == 'random_edges':
                # Degree-preserving edge ablation: keep sources, edge attributes,
                # and the multiset of targets, but break their original pairings
                # deterministically inside each connected image component.
                generator = torch.Generator().manual_seed(
                    int(os.environ.get('GRAPHGAT_RANDOM_EDGE_SEED', '2026')) + component
                )
                permutation = torch.randperm(edges.shape[1], generator=generator)
                edges = torch.stack((edges[0], edges[1, permutation]))
            node_types = compact['node_type'][nodes]
            roots = nodes[node_types == 1]
            assert roots.numel() == 1
            node_chunks.append(nodes)
            edge_chunks.append(edges)
            relation_chunks.append(compact['edge_relation_type'][es:ee])
            reverse_chunks.append(compact['edge_is_reverse'][es:ee])
            batch_chunks.append(torch.full((nodes.numel(),), batch_id, dtype=torch.long))
            root_globals.append(roots.item())

        node_global = torch.cat(node_chunks)
        edge_global = torch.cat(edge_chunks, dim=1)
        global_to_local[node_global] = torch.arange(node_global.numel())
        edge_local = global_to_local[edge_global]
        root_local = global_to_local[torch.tensor(root_globals)]
        assert edge_local.min().item() >= 0

        node_types = compact['node_type'][node_global].long()
        feature_rows = compact['node_feature_row'][node_global]
        x = torch.empty((node_global.numel(), 512), dtype=torch.float32)
        image_mask = node_types == 1
        x[image_mask] = image_features[feature_rows[image_mask]]
        x[~image_mask] = entity_features[feature_rows[~image_mask]]
        edge_attr = relation_type_features[torch.cat(relation_chunks)]
        edge_reverse = torch.cat(reverse_chunks).float().unsqueeze(1)
        node_batch = torch.cat(batch_chunks)
        global_to_local[node_global] = -1

        return {
            'x': x.to(DEVICE),
            'node_type': node_types.to(DEVICE),
            'edge_index': edge_local.to(DEVICE),
            'edge_attr': edge_attr.to(DEVICE),
            'edge_reverse': edge_reverse.to(DEVICE),
            'batch': node_batch.to(DEVICE),
            'root_local': root_local.to(DEVICE),
            'components': component_rows,
        }

    sample_batch = build_component_batch([0, 1, 2])
    print('Sample nodes:', sample_batch['x'].shape)
    print('Sample edges:', sample_batch['edge_index'].shape)
    print('Sample image roots:', sample_batch['root_local'].shape)





    EDGE_MODE = os.environ.get('GRAPHGAT_EDGE_MODE', 'real_edges').strip().lower()
    if EDGE_MODE not in {'real_edges', 'no_edge', 'random_edges'}:
        raise ValueError(
            'GRAPHGAT_EDGE_MODE must be real_edges, no_edge, or random_edges, '
            f'got {EDGE_MODE!r}'
        )
    model = EdgeAwareGAT(edge_mode=EDGE_MODE).to(DEVICE)
    with torch.no_grad():
        sample_graph_embeddings = model.encode_graph(sample_batch)
    assert sample_graph_embeddings.shape == (3, 128)
    print('GAT forward pass: OK', sample_graph_embeddings.shape)


    # ## Training input and audit status
    # Only writes inside experiment_v2. `RUN_TRAINING=False` by default to prevent unintentional full GPU training. Enable only after preparation and review.



    from .guard import verify_frozen_val_test,sha256
    from .audit import audit_pool
    print('Validating frozen eval artifacts and v2 query pool...')
    verify_frozen_val_test(VG_ROOT,full_sha=True)
    local_split_file=VG_ROOT/'processed/splits_v2.parquet'
    if local_split_file.is_file():
        assert audit_pool(VG_ROOT,EXP_ROOT)['status']=='PASS'
    else:
        print('WARNING: local processed/splits_v2.parquet is absent; direct split-file audit is NOT VERIFIED locally.')
        assert audit_pool(VG_ROOT,EXP_ROOT,skip_split_checks=True)['status']=='PASS'
    base_neg_manifest_path=EXP_ROOT/'data/hard_negatives/manifest.json'
    full_manifest_path=EXP_ROOT/'data/full_query_pool/handoff_manifest.json'
    integrity_path=EXP_ROOT/'data/full_query_pool/integrity_report.json'
    for required in [base_neg_manifest_path, full_manifest_path, integrity_path,
                     EXP_ROOT/'data/embeddings/manifest.json']:
        assert required.is_file(), f'Missing required full-pool artifact: {required}'
    base_neg_manifest=json.loads(base_neg_manifest_path.read_text())
    full_manifest=json.loads(full_manifest_path.read_text())
    integrity=json.loads(integrity_path.read_text())
    FULL_QUERY_COUNT=4_184_397
    assert base_neg_manifest['queries']==1_670_618 and base_neg_manifest['negative_count']==50
    assert full_manifest['status']=='TRAINING_DATA_READY'
    assert full_manifest['total_unique_train_queries']==FULL_QUERY_COUNT
    assert full_manifest['delta_query_count']==2_513_779
    assert integrity['status']=='PASS' and integrity['all_unique_queries']==FULL_QUERY_COUNT
    assert integrity['eval_overlap']==0 and integrity['negative_count_per_query']==50
    RUN_NAME = (
        'gat_full_pool_sample5_v2'
        if EDGE_MODE == 'real_edges'
        else f'gat_full_pool_sample5_v2_{EDGE_MODE}'
    )
    SUBSET_TRAIN_IMAGES = int(os.environ.get('GRAPHGAT_SUBSET_TRAIN_IMAGES', '0'))
    QUERIES_PER_IMAGE = int(os.environ.get('GRAPHGAT_QUERIES_PER_IMAGE', '5'))
    SKIP_VALIDATION = os.environ.get('GRAPHGAT_SKIP_VALIDATION', '0') == '1'
    if SUBSET_TRAIN_IMAGES:
        RUN_NAME += f'_subset{SUBSET_TRAIN_IMAGES}_q{QUERIES_PER_IMAGE}_benchmark'
    RUN_NAME = os.environ.get('GRAPHGAT_RUN_NAME', RUN_NAME)
    RUN_ROOT=GUARD.mkdir(MODEL_ROOT/RUN_NAME)
    seed_text = os.environ.get('GRAPHGAT_SEEDS', '2026')
    RUN_SEEDS = [int(value.strip()) for value in seed_text.split(',') if value.strip()]
    if not RUN_SEEDS:
        raise ValueError('GRAPHGAT_SEEDS must contain at least one integer seed')
    if len(RUN_SEEDS) != len(set(RUN_SEEDS)):
        raise ValueError(f'GRAPHGAT_SEEDS contains duplicates: {RUN_SEEDS}')
    cfg={'train_negatives':50,'queries_per_image_per_epoch':QUERIES_PER_IMAGE,
         'query_batch_size':8,
         'epochs':int(os.environ.get(
             'GRAPHGAT_EPOCHS', '100' if EDGE_MODE == 'real_edges' else '50'
         )),
         'validation_queries':520370,'seeds':RUN_SEEDS, 'checkpoint_every_batches':500}
    ALPHA_GRID=[round(x/10,1) for x in range(11)]
    RUN_SMOKE_TEST = os.environ.get('GRAPHGAT_SMOKE_TEST', '0') == '1'
    RUN_TRAINING = os.environ.get('GRAPHGAT_RUN_TRAINING', '1') == '1'
    print('Run configuration:', {
        'run_name': RUN_NAME,
        'edge_mode': EDGE_MODE,
        'epochs': cfg['epochs'],
        'smoke_test': RUN_SMOKE_TEST,
        'training': RUN_TRAINING,
    })

    # Keep old graph audit status as data: don't forge human approval.
    audit_status=json.loads((GRAPH_ROOT/'graph_manifest.json').read_text()).get('manual_audit_status')
    SEMANTIC_AUDIT_CONFIRMED=False  # Only change True after genuine team review.
    if audit_status not in {'complete','approved'} and not SEMANTIC_AUDIT_CONFIRMED:
        print('WARNING: HUMAN SEMANTIC GRAPH AUDIT PENDING. Training can be exploratory only.')

    # Build exact text_id -> tensor row mapping. V2 tensor shards are sorted by index filename.
    # ============================================
    # Load V2 text embeddings with progress
    # ============================================

    embedding_sources = [
        ('base', EXP_ROOT/'data/embeddings/indexes',
                 EXP_ROOT/'data/embeddings/shards'),
        ('delta', EXP_ROOT/'data/full_query_pool/embeddings_delta/indexes',
                  EXP_ROOT/'data/full_query_pool/embeddings_delta/shards'),
    ]
    indexed_shards = [
        (label, index_path, shard_root/f'{index_path.stem}.pt')
        for label, index_root, shard_root in embedding_sources
        for index_path in sorted(index_root.glob('part_*.parquet'))
    ]

    assert indexed_shards, 'No full-pool embedding indexes found'
    assert sum(label=='base' for label,_,_ in indexed_shards)==102
    assert sum(label=='delta' for label,_,_ in indexed_shards)==154
    assert all(shard_path.is_file() for _,_,shard_path in indexed_shards)

    idx_frames = []
    offset = 0
    train_text_embeddings = torch.empty(
        (FULL_QUERY_COUNT, 512), dtype=torch.float16
    )

    total_shards = len(indexed_shards)

    log_status(
        f"[EMBEDDINGS] Found {total_shards} shards. Starting validation and loading..."
    )

    t0 = time.perf_counter()

    for source_label, index_path, shard_path in tqdm(
        indexed_shards,
        desc="Loading text embeddings",
        unit="shard",
        dynamic_ncols=True,
        mininterval=2
    ):
        part_index = pd.read_parquet(
            index_path, columns=['text_id','embedding_row']
        )

        part_tensor = torch.load(
            shard_path,
            map_location='cpu',
            weights_only=True
        )

        assert len(part_index) == len(part_tensor)
        assert part_index.embedding_row.tolist() == list(
            range(len(part_index))
        )

        part_index = part_index.copy()

        part_index['embedding_row'] = (
            part_index.embedding_row + offset
        )

        idx_frames.append(part_index)
        train_text_embeddings[offset:offset+len(part_tensor)].copy_(part_tensor)

        offset += len(part_tensor)

    log_status(
        f"[EMBEDDINGS] All {total_shards} shards loaded "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    # Build text ID mapping
    log_status("[EMBEDDINGS] Building text ID index...")

    train_text_index = pd.concat(
        idx_frames,
        ignore_index=True
    )

    assert train_text_index.text_id.is_unique
    assert len(train_text_index) == FULL_QUERY_COUNT

    text_row_by_id = dict(
        zip(
            train_text_index.text_id.astype(str),
            train_text_index.embedding_row
        )
    )
    del train_text_index

    assert offset == FULL_QUERY_COUNT
    del idx_frames

    log_status(
        f"[EMBEDDINGS] COMPLETED | "
        f"Shape: {tuple(train_text_embeddings.shape)} | "
        f"Total queries: {FULL_QUERY_COUNT:,}"
    )




    def encode_image_components(model, image_ids, component_batch_size=128):
        components = [image_to_component[x] for x in image_ids]
        outputs = []
        model.eval()
        with torch.no_grad():
          for start in tqdm(
              range(0, len(components), component_batch_size),
              desc="Validation | Encoding graph images",
              unit="batch",
              dynamic_ncols=True,
              mininterval=2
          ):
                graph_batch = build_component_batch(
                    components[start:start + component_batch_size]
                )
                outputs.append(model.encode_graph(graph_batch).cpu())
        return torch.cat(outputs)

    _val_cache = None

    def get_val_cache(max_queries, seed):
        nonlocal _val_cache
        if _val_cache is None:
            val_root = CLIP_ROOT / 'val'
            candidate_df = pd.read_parquet(
                val_root / 'candidate_images.parquet', columns=['image_id']
            )
            candidate_ids = candidate_df.image_id.astype(str).tolist()
            result_table = pq.read_table(
                val_root / 'retrieval_results.parquet',
                columns=[
                    'text_id', 'best_positive_rank', 'topk_image_ids',
                    'topk_scores', 'positive_image_id'
                ],
            )
            if max_queries < len(result_table):
              rng = np.random.default_rng(seed)
              selected = np.sort(rng.choice(len(result_table), max_queries, replace=False))
              result_table = result_table.take(selected)
            result_df = result_table.to_pandas()
            text_index = pd.read_parquet(val_root / 'text_embedding_index.parquet')
            text_row = dict(zip(text_index.text_id.astype(str), text_index.embedding_row))
            text_features = torch.load(
                val_root / 'text_embeddings.pt', map_location='cpu', weights_only=True
            ).float()
            _val_cache = (candidate_ids, result_df, text_row, text_features)
        return _val_cache

    def evaluate_validation(model, max_queries, seed):

        val_start_time = time.perf_counter()

        log_status("[VAL] Starting validation...")

        candidate_ids, result_df, text_row, text_features = get_val_cache(
            max_queries, seed
        )

        log_status(
            f"[VAL] Loaded {len(result_df):,} queries "
            f"and {len(candidate_ids):,} candidate images"
        )

        graph_embeddings = encode_image_components(model, candidate_ids)
        graph_row = {image_id: i for i, image_id in enumerate(candidate_ids)}

        totals = {alpha: {'r1': 0, 'r5': 0, 'r10': 0, 'r50': 0, 'rr': 0.0}
                  for alpha in ALPHA_GRID}
        model.eval()
        eval_batch_size = 512
        with torch.no_grad():
          for start in tqdm(
              range(0, len(result_df), eval_batch_size),
              desc="Validation | Scoring queries",
              unit="batch",
              dynamic_ncols=True,
              mininterval=2
          ):
                rows = result_df.iloc[start:start + eval_batch_size]
                q_rows = [text_row[x] for x in rows.text_id]
                query_z = model.encode_query(text_features[q_rows].to(DEVICE)).cpu()
                for local_i, row in enumerate(rows.itertuples(index=False)):
                    ids = list(row.topk_image_ids)
                    clip_scores = torch.tensor(row.topk_scores, dtype=torch.float32)
                    candidate_z = graph_embeddings[[graph_row[x] for x in ids]]
                    graph_scores = query_z[local_i] @ candidate_z.T
                    clip_z = (clip_scores - clip_scores.mean()) / clip_scores.std().clamp_min(1e-6)
                    graph_z = (graph_scores - graph_scores.mean()) / graph_scores.std().clamp_min(1e-6)
                    try:
                        positive_position = ids.index(row.positive_image_id)
                    except ValueError:
                        positive_position = None
                    for alpha in ALPHA_GRID:
                        if positive_position is None:
                            rank = int(row.best_positive_rank)
                        else:
                            fused = alpha * clip_z + (1.0 - alpha) * graph_z
                            positive_score = fused[positive_position]
                            rank = 1 + int((fused > positive_score).sum())
                        item = totals[alpha]
                        item['r1'] += rank <= 1
                        item['r5'] += rank <= 5
                        item['r10'] += rank <= 10
                        item['r50'] += rank <= 50
                        item['rr'] += 1.0 / rank

        n = len(result_df)
        metrics = {}
        for alpha, item in totals.items():
            metrics[str(alpha)] = {
                'Recall@1': item['r1'] / n,
                'Recall@5': item['r5'] / n,
                'Recall@10': item['r10'] / n,
                'Recall@50': item['r50'] / n,
                'MRR': item['rr'] / n,
                'num_queries': n,
            }

        best_alpha = max(
            ALPHA_GRID,
            key=lambda a: metrics[str(a)]['MRR']
        )

        elapsed = time.perf_counter() - val_start_time

        log_status(
            f"[VAL] COMPLETED | "
            f"Time: {elapsed / 60:.1f} min | "
            f"Best alpha: {best_alpha} | "
            f"MRR: {metrics[str(best_alpha)]['MRR']:.6f}"
        )

        return best_alpha, metrics




    def train_one_seed_v2(seed, cfg):
        smoke_limit=cfg.get('smoke_test_batches')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        fresh_model = EdgeAwareGAT(edge_mode=EDGE_MODE).to(DEVICE)
        optimizer = torch.optim.AdamW(
            fresh_model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        seed_dir = GUARD.mkdir(RUN_ROOT / f'seed_{seed}')
        resume_path = seed_dir / 'last_checkpoint.pt'

        import duckdb
        from .sampler import EpochQuerySampler
        from .guard import sha256
        query_files = [
            EXP_ROOT/'data/query_pool/queries.parquet',
            EXP_ROOT/'data/full_query_pool/query_index/delta_queries.parquet',
        ]
        negative_globs = [
            EXP_ROOT/'data/hard_negatives/parts/part_*.parquet',
            EXP_ROOT/'data/full_query_pool/hard_negatives_delta/parts/part_*.parquet',
        ]
        assert all(path.is_file() for path in query_files)
        assert all(list(path.parent.glob(path.name)) for path in negative_globs)
        query_pool = pd.concat([
            pd.read_parquet(path, columns=['positive_image_id','text_id'])
            for path in query_files
        ], ignore_index=True)
        assert len(query_pool)==FULL_QUERY_COUNT
        assert query_pool.text_id.is_unique
        assert query_pool.positive_image_id.nunique()==83_540
        if SUBSET_TRAIN_IMAGES:
            split_frame = pd.read_parquet(
                PROCESSED_ROOT/'splits_v2.parquet',
                columns=['image_id', 'split', 'assignment_key'],
            )
            subset_ids = set(
                split_frame.loc[split_frame.split == 'train']
                .sort_values(['assignment_key', 'image_id'])
                .head(SUBSET_TRAIN_IMAGES).image_id.astype(str)
            )
            assert len(subset_ids) == SUBSET_TRAIN_IMAGES
            query_pool = query_pool[
                query_pool.positive_image_id.astype(str).isin(subset_ids)
            ].reset_index(drop=True)
            assert query_pool.positive_image_id.nunique() == SUBSET_TRAIN_IMAGES
            log_status(
                f"[SUBSET] Fixed train images: {SUBSET_TRAIN_IMAGES:,} | "
                f"available queries: {len(query_pool):,}"
            )
        expected_train_images = query_pool.positive_image_id.nunique()
        sampler = EpochQuerySampler(
            query_pool.itertuples(index=False,name=None), seed=seed,
            per_image=cfg['queries_per_image_per_epoch']
        )
        del query_pool
        pool_sha=':'.join(sha256(path) for path in query_files)
        def epoch_training_frame(epoch):

            t0 = time.perf_counter()

            log_status(
                f"[EPOCH {epoch+1}] "
                "Starting full-pool query sampling..."
            )

            choices = sampler.sample(epoch)

            choices_df = pd.DataFrame(
                choices,
                columns=['positive_image_id', 'text_id']
            )

            log_status(
                f"[EPOCH {epoch+1}] "
                f"Sampled {len(choices_df):,} queries"
            )

            log_status(
                f"[EPOCH {epoch+1}] "
                "Joining queries with hard-negative parts..."
            )

            cx = duckdb.connect(database=':memory:')

            try:
                cx.register('chosen_v2', choices_df)

                parts_sql = ','.join(
                    f"'{str(path).replace(chr(39), chr(39)*2)}'"
                    for path in negative_globs
                )

                frame = cx.execute(f"""
                    SELECT
                        h.text_id,
                        h.positive_image_id,
                        h.negative_image_ids
                    FROM read_parquet([{parts_sql}]) h
                    JOIN chosen_v2 s
                        ON h.text_id = s.text_id
                        AND h.positive_image_id = s.positive_image_id
                    ORDER BY h.text_id
                """).fetchdf()

            finally:
                cx.close()

            assert len(choices_df)==expected_train_images*cfg['queries_per_image_per_epoch']
            assert len(frame) == len(choices_df)
            assert frame.text_id.is_unique

            log_status(
                f"[EPOCH {epoch+1}] "
                f"Training data ready | "
                f"{len(frame):,} queries | "
                f"{time.perf_counter() - t0:.1f}s"
            )

            return frame
        rng = np.random.default_rng(seed + 10_000)
        history, best_mrr = [], -1.0
        start_epoch, resume_start, resume_order = 0, 0, None
        resume_losses, resume_correct, resume_seen = [], 0, 0

        if resume_path.exists() and smoke_limit is None:
            state = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            assert state.get('architecture') == 'GATConv', 'Checkpoint architecture mismatch'
            assert state.get('edge_mode', 'real_edges') == EDGE_MODE, 'Checkpoint edge mode mismatch'
            assert state.get('v2_pool_sha256')==pool_sha, 'Cannot resume with a changed v2 query pool'
            assert state.get('seed')==seed
            if state.get('complete') and state['epoch'] >= cfg['epochs']:
                print(f'Seed {seed} already complete')
                return state['history']
            fresh_model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            start_epoch = state['epoch']
            resume_start = state['next_start']
            resume_order = state['order']
            resume_losses = state['losses']
            resume_correct = state['correct']
            resume_seen = state['seen']
            history = state['history']
            best_mrr = state['best_mrr']
            rng.bit_generator.state = state['numpy_generator_state']
            random.setstate(state['python_random_state'])
            np.random.set_state(state['numpy_random_state'])
            torch.set_rng_state(state['torch_rng_state'].cpu())
            if torch.cuda.is_available() and state.get('cuda_rng_states') is not None:
                torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng_states']])
            print(f'Resuming seed {seed}: epoch {start_epoch + 1}, query {resume_start:,}')

        def save_resume(epoch, next_start, order, losses, correct, seen, complete=False):
            tmp_resume = resume_path.with_name(resume_path.name + '.tmp')
            torch.save({
                'v2_pool_sha256':pool_sha,
                'model_state_dict': fresh_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'architecture': 'GATConv',
                'edge_mode': EDGE_MODE,
                'seed': seed, 'epoch': epoch, 'next_start': next_start,
                'order': order, 'losses': losses, 'correct': correct, 'seen': seen,
                'history': history, 'best_mrr': best_mrr, 'complete': complete,
                'numpy_generator_state': rng.bit_generator.state,
                'python_random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': (torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available() else None),
            }, tmp_resume)
            tmp_resume.replace(resume_path)

        for epoch in range(start_epoch, cfg['epochs']):

            epoch_start_time = time.perf_counter()

            log_status(
                f"========== EPOCH {epoch+1}/{cfg['epochs']} =========="
            )

            train_df = epoch_training_frame(epoch)
            print(f'Epoch {epoch+1}: sampled {len(train_df)} distinct queries')
            if epoch == start_epoch and resume_order is not None:
                order = resume_order
                batch_start = resume_start
                losses, correct, seen = resume_losses, resume_correct, resume_seen
            else:
                order = rng.permutation(len(train_df))
                batch_start = 0
                losses, correct, seen = [], 0, 0
            fresh_model.train()
            starts = range(batch_start, len(order), cfg['query_batch_size'])
            progress = tqdm(
                starts, total=(len(order) + cfg['query_batch_size'] - 1) // cfg['query_batch_size'],
                initial=batch_start // cfg['query_batch_size'],
                desc=f'Seed {seed} | Epoch {epoch + 1}/{cfg["epochs"]}',
            )
            for batch_number, start in enumerate(progress, start=batch_start // cfg['query_batch_size']):
                rows = train_df.iloc[order[start:start + cfg['query_batch_size']]]
                candidate_lists, query_rows, all_components = [], [], []
                for row in rows.itertuples(index=False):
                    negatives = list(row.negative_image_ids)
                    chosen = rng.choice(
                        len(negatives), cfg['train_negatives'], replace=False
                    )
                    image_ids = [row.positive_image_id] + [negatives[i] for i in chosen]
                    components = [image_to_component[str(x)] for x in image_ids]
                    candidate_lists.append(components)
                    all_components.extend(components)
                    query_rows.append(text_row_by_id[row.text_id])

                unique_components = list(dict.fromkeys(all_components))
                graph_batch = build_component_batch(unique_components)
                graph_z = fresh_model.encode_graph(graph_batch)
                component_position = {x: i for i, x in enumerate(unique_components)}
                query_z = fresh_model.encode_query(
                    train_text_embeddings[query_rows].to(DEVICE).float()
                )
                logits = []
                for query_i, components in enumerate(candidate_lists):
                    positions = torch.tensor(
                        [component_position[x] for x in components], device=DEVICE
                    )
                    logits.append(query_z[query_i] @ graph_z[positions].T)
                logits = torch.stack(logits) * fresh_model.logit_scale.exp().clamp(max=100)
                target = torch.zeros(len(logits), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fresh_model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
                correct += int((logits.argmax(1) == 0).sum())
                seen += len(logits)
                progress.set_postfix(loss=f'{np.mean(losses[-50:]):.4f}', top1=f'{correct/seen:.3f}')
                if smoke_limit is not None and len(losses)>=smoke_limit:
                    break
                if (batch_number + 1) % cfg['checkpoint_every_batches'] == 0:

                    save_resume(
                        epoch,
                        start + len(rows),
                        order,
                        losses,
                        correct,
                        seen
                    )

                    log_status(
                        f"[CHECKPOINT] "
                        f"Epoch {epoch+1}/{cfg['epochs']} | "
                        f"Batch {batch_number+1} | "
                        f"Queries processed: {seen:,}/{len(train_df):,} | "
                        f"Saved successfully"
                    )
            if smoke_limit is not None:
                smoke_result={
                    'status':'PASS', 'batches':len(losses), 'queries':seen,
                    'mean_loss':float(np.mean(losses)), 'top1':correct/seen,
                }
                print('FULL-POOL SMOKE TEST:',json.dumps(smoke_result,indent=2))
                return smoke_result
            save_resume(
                epoch, len(order), order, losses, correct, seen
            )

            log_status(
                f"[TRAIN] Epoch {epoch+1} completed | "
                f"Loss: {np.mean(losses):.5f} | "
                f"Top1: {correct/seen:.4f}"
            )

            if SKIP_VALIDATION:
                elapsed = time.perf_counter() - epoch_start_time
                benchmark = {
                    'epoch': epoch + 1,
                    'train_images': expected_train_images,
                    'queries_per_image': cfg['queries_per_image_per_epoch'],
                    'queries': seen,
                    'batches': len(losses),
                    'train_loss': float(np.mean(losses)),
                    'train_top1': correct / seen,
                    'elapsed_seconds': elapsed,
                    'seconds_per_batch': elapsed / len(losses),
                    'validation_skipped': True,
                }
                history.append(benchmark)
                GUARD.save_json(seed_dir / 'history.json', history)
                save_resume(epoch + 1, 0, None, [], 0, 0, complete=True)
                print('SUBSET EPOCH BENCHMARK:', json.dumps(benchmark, indent=2))
                continue

            log_status(
                f"[TRAIN] Starting validation for epoch {epoch+1}..."
            )

            best_alpha, val_metrics = evaluate_validation(
                fresh_model,
                cfg['validation_queries'],
                seed
            )
            best_metrics = val_metrics[str(best_alpha)]
            clip_metrics = val_metrics['1.0']
            val_mrr = best_metrics['MRR']
            metric_names = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@50', 'MRR']
            improvement = {
                name: best_metrics[name] - clip_metrics[name] for name in metric_names
            }
            record = {
                'epoch': epoch + 1,
                'train_loss': float(np.mean(losses)),
                'train_top1': correct / seen,
                'best_alpha': best_alpha,
                'clip_only': clip_metrics,
                'clip_plus_graph': best_metrics,
                'improvement': improvement,
            }
            history.append(record)
            print(json.dumps(record, indent=2))
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_path = seed_dir / 'best_gat.pt'
                tmp_best = best_path.with_name(best_path.name + '.tmp')
                torch.save({
                    'model_state_dict': fresh_model.state_dict(),
                    'seed': seed, 'epoch': epoch + 1,
                    'alpha': best_alpha, 'val_metrics': val_metrics,
                    'run_name': RUN_NAME, 'architecture': 'GATConv',
                    'edge_mode': EDGE_MODE,
                    'split_version': 'split_v2',
                    'manifest_audit_status': audit_status,
                    'semantic_audit_confirmed': SEMANTIC_AUDIT_CONFIRMED,
                    'v2_pool_sha256':pool_sha,
                    'query_pool_mode':'full_base_plus_delta_sample5',
                    'full_query_count':FULL_QUERY_COUNT,
                }, tmp_best)
                tmp_best.replace(best_path)
            GUARD.save_json(seed_dir / 'history.json', history)
            save_resume(
                epoch + 1, 0, None, [], 0, 0,
                complete=(epoch + 1 == cfg['epochs']),
            )
            # Immutable per-epoch recovery point. last_checkpoint.pt continues to
            # advance during the next epoch, while this snapshot is never reused
            # or overwritten by another epoch.
            epoch_checkpoint = seed_dir / f'checkpoint_epoch_{epoch + 1:03d}.pt'
            epoch_checkpoint_tmp = epoch_checkpoint.with_suffix('.pt.tmp')
            shutil.copy2(resume_path, epoch_checkpoint_tmp)
            epoch_checkpoint_tmp.replace(epoch_checkpoint)
            resume_order = None
            log_status(
              f"[EPOCH {epoch+1}] FINISHED | "
              f"Elapsed: {(time.perf_counter() - epoch_start_time) / 60:.1f} min | "
              f"Validation MRR: {val_mrr:.6f} | "
              f"Best MRR so far: {best_mrr:.6f}"
            )
        return history

    run_histories = {}
    if RUN_SMOKE_TEST:
        smoke_cfg={**cfg,'epochs':1,'smoke_test_batches':10}
        run_histories['smoke']=train_one_seed_v2(cfg['seeds'][0],smoke_cfg)
    elif RUN_TRAINING:
        for run_seed in cfg['seeds']:
            run_histories[str(run_seed)] = train_one_seed_v2(run_seed, cfg)
        print('Training complete:', RUN_ROOT)
    else:
        print('Dry run: set RUN_TRAINING=True only after running preparation notebook and reviewing estimated GPU cost.')


    # ## Output
    # Historical baseline remains in `dataset/visual_genome/model/gat_v1`. All new GAT checkpoints and JSON histories are in `experiment_v2/model/gat_dynamic_v2/`. Model selection uses original **validation only**. Do not tune on test.


if __name__ == "__main__":
    main()
