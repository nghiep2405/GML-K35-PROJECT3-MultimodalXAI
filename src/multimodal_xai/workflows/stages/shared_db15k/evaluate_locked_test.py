# Uses the existing notebook variables/helpers; does not train or tune on test.
import copy
from datetime import datetime, timezone
from transformers import CLIPModel, CLIPTokenizerFast

assert (RUN / 'complete.json').exists(), 'Finish the configured training run before final test.'
best_path = RUN / 'best_adamf.pt'
saved = torch.load(best_path, map_location='cpu', weights_only=False)
assert saved['identity'] == IDENTITY, 'Checkpoint/data identity mismatch'
compatible_config(saved['cfg'], CFG)
alpha = float(saved['alpha_clip'])  # LOCKED validation alpha; no grid search here
assert 0. <= alpha <= 1.
print('LOCKED: epoch =', saved['epoch'], '| alpha_clip =', alpha)

# Re-read original files because earlier cells removed test rows from memory.
for key in ['queries', 'relevance', 'candidate_pools']:
    assert sha(paths[key]) == HASHES[key], f'Input changed: {key}'
test_q = pd.read_parquet(paths['queries'])
test_q = test_q.loc[test_q['split'].eq('test')].sort_values('query_id').reset_index(drop=True)
assert len(test_q) and test_q.query_id.is_unique, 'Missing or duplicate test queries'
assert test_q.prediction_direction.isin(['direct', 'head', 'tail']).all()
assert not test_q.query_id.isin(queries.query_id).any(), 'Query split overlap'
all_pools = pd.read_parquet(paths['candidate_pools'])
test_images = all_pools.loc[all_pools['split'].eq('test'), 'image_id']
assert len(test_images) and test_images.is_unique
assert not test_images.isin(all_pools.loc[all_pools['split'].isin(['train', 'val']), 'image_id']).any(), 'Candidate split overlap'
assert test_images.isin(row_map.index).all()
test_pool = np.sort(test_images.map(row_map).to_numpy(np.int64))
test_rel = pd.read_parquet(paths['relevance'])
test_rel = test_rel.loc[test_rel.query_id.isin(test_q.query_id)].copy()
assert test_rel.image_id.isin(test_images).all(), 'Relevant image outside test pool'
test_rel['feature_row'] = test_rel.image_id.map(row_map)
test_positives = test_rel.groupby('query_id').feature_row.apply(
    lambda x: np.unique(x.to_numpy(np.int64))).to_dict()
assert set(test_positives) == set(test_q.query_id), 'Missing test ground truth'

# Use exactly the text encoder revision already pinned on validation.
tokenizer_test = CLIPTokenizerFast.from_pretrained(
    encoder_meta['name'], revision=encoder_meta['revision'])
encoder_test = CLIPModel.from_pretrained(
    encoder_meta['name'], revision=encoder_meta['revision']).to(DEVICE).eval()
encoder_test.requires_grad_(False)
text_outputs = []
with torch.no_grad():
    for start in tqdm(range(0, len(test_q), 128), desc='Encode test queries'):
        tokens = tokenizer_test(
            test_q.query_text.iloc[start:start+128].tolist(),
            padding=True, truncation=True, max_length=77,
            return_tensors='pt').to(DEVICE)
        features = encoder_test.text_projection(
            encoder_test.text_model(**tokens, return_dict=True).pooler_output)
        text_outputs.append(F.normalize(features.float(), dim=1).cpu())
test_text_z = torch.cat(text_outputs)
del encoder_test, tokenizer_test, text_outputs, tokens, features
torch.cuda.empty_cache()

# Copy model to keep the original training model/optimizer untouched.
test_model = copy.deepcopy(model).to(DEVICE)
test_model.load_state_dict(saved['model'])
test_model.eval().requires_grad_(False)
pool_features = F.normalize(
    torch.tensor(np.array(image_array[test_pool], copy=True)).float(), dim=1).to(DEVICE)
pool_position = {int(row): i for i, row in enumerate(test_pool)}
records = []
with torch.no_grad():
    joint = torch.cat([test_model.joint(torch.arange(s, min(s+1024, N), device=DEVICE))
                       for s in range(0, N, 1024)])
    for start in tqdm(range(0, len(test_q), 32), desc='Test CLIP + locked AdaMF'):
        scores = test_text_z[start:start+32].to(DEVICE) @ pool_features.T
        for local_i, row in enumerate(test_q.iloc[start:start+32].itertuples(index=False)):
            clip_scores = scores[local_i]
            positives = test_positives[row.query_id]
            positive_positions = torch.tensor(
                [pool_position[int(x)] for x in positives], device=DEVICE)
            clip_rank = 1 + int((clip_scores > clip_scores[positive_positions].max()).sum())
            order_test = torch.argsort(clip_scores, descending=True, stable=True)[:min(CFG['topk'], len(test_pool))]
            candidate_rows = torch.tensor(test_pool[order_test.cpu().numpy()], device=DEVICE)
            hybrid_rank = clip_rank
            if row.prediction_direction != 'direct' and alpha != 1.:
                # Scorer uses anchor/relation/direction and candidate entity IDs only.
                ids = IMAGE_ENTITIES[candidate_rows.cpu()].to(DEVICE)
                anchor, relation = int(row.anchor_entity_index), int(row.relation_index)
                assert 0 <= anchor < N and 0 <= relation < R
                candidate_z = joint[ids]
                anchor_z = joint[anchor].expand_as(candidate_z)
                relation_z = test_model.rel_embeddings(torch.full_like(ids, relation))
                h, t = ((anchor_z, candidate_z) if row.prediction_direction == 'tail'
                        else (candidate_z, anchor_z))
                graph_scores = test_model.margin - test_model._calc(h, t, relation_z, 'normal')
                assert torch.isfinite(graph_scores).all()
                fused = alpha*zscore(clip_scores[order_test]) + (1.-alpha)*zscore(graph_scores)
                # Labels are used ONLY below to evaluate the already-computed scores.
                positive_mask = torch.isin(candidate_rows, torch.tensor(positives, device=DEVICE))
                if positive_mask.any():
                    hybrid_rank = 1 + int((fused > fused[positive_mask].max()).sum())
            records.append(dict(query_id=row.query_id, direction=row.prediction_direction,
                                clip_rank=clip_rank, hybrid_rank=hybrid_rank))

test_ranks = pd.DataFrame(records)
masks = {'combined': np.ones(len(test_ranks), dtype=bool),
         'direct': test_ranks.direction.eq('direct').to_numpy(),
         'indirect': test_ranks.direction.ne('direct').to_numpy(),
         'head': test_ranks.direction.eq('head').to_numpy(),
         'tail': test_ranks.direction.eq('tail').to_numpy()}
result_rows = []
for method in ['clip', 'hybrid']:
    for group, mask in masks.items():
        result_rows.append(dict(method=method, group=group,
            epoch=saved['epoch'], alpha_clip=1. if method=='clip' else alpha,
            **rank_metrics(test_ranks.loc[mask, method+'_rank'].to_numpy())))
test_results = pd.DataFrame(result_rows)
TEST_OUT = RUN / ('test_locked_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
TEST_OUT.mkdir(exist_ok=False)
test_results.to_csv(TEST_OUT / 'test_metrics.csv', index=False)
test_ranks.to_csv(TEST_OUT / 'test_query_ranks.csv', index=False)
write_json(dict(checkpoint=str(best_path), checkpoint_sha256=sha(best_path),
                selected_validation_epoch=saved['epoch'], locked_alpha_clip=alpha,
                seed=saved['cfg']['seed'], topk=CFG['topk'],
                encoder=encoder_meta, input_hashes=HASHES,
                num_queries=len(test_q), num_candidates=len(test_pool),
                protocol='Same optimistic rank / best-positive / top-k fallback as validation; direct uses CLIP.'),
           TEST_OUT / 'test_protocol.json')
display(test_results)
print('TEST outputs:', TEST_OUT)
print('No training or alpha search was performed on test.')
del test_model, pool_features, joint, saved
torch.cuda.empty_cache()
