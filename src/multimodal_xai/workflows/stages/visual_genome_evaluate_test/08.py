def encode_image_components(model, image_ids, component_batch_size=128):
    components = [image_to_component[x] for x in image_ids]
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(components), component_batch_size):
            graph_batch = build_component_batch(
                components[start:start + component_batch_size]
            )
            outputs.append(model.encode_graph(graph_batch).cpu())
    return torch.cat(outputs)

def evaluate_frozen_test(model, max_queries, seed):
    test_root = CLIP_ROOT / 'test'
    candidate_df = pd.read_parquet(
        test_root / 'candidate_images.parquet', columns=['image_id']
    )
    candidate_ids = candidate_df.image_id.astype(str).tolist()
    graph_embeddings = encode_image_components(model, candidate_ids)
    graph_row = {image_id: i for i, image_id in enumerate(candidate_ids)}

    result_table = pq.read_table(
        test_root / 'retrieval_results.parquet',
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
    text_index = pd.read_parquet(test_root / 'text_embedding_index.parquet')
    text_row = dict(zip(text_index.text_id.astype(str), text_index.embedding_row))
    text_features = torch.load(
        test_root / 'text_embeddings.pt', map_location='cpu', weights_only=True
    ).float()

    totals = {alpha: {'r1': 0, 'r5': 0, 'r10': 0, 'r50': 0, 'rr': 0.0}
              for alpha in ALPHA_GRID}
    model.eval()
    eval_batch_size = 512
    with torch.no_grad():
        for start in range(0, len(result_df), eval_batch_size):
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
    best_alpha = max(ALPHA_GRID, key=lambda a: metrics[str(a)]['MRR'])
    return best_alpha, metrics
