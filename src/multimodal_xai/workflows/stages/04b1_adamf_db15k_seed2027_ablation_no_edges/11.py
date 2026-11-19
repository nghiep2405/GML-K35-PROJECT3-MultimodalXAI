# No triples => no KGC objective. Explicit CLIP-only graph-removal control.
IDENTITY = dict(version='db15k-no-edges-v1', hashes=HASHES, graph_hash=GRAPH_HASH,
    model_hash=MODEL_HASH, cache_identity=cache_identity,
    evaluation_hash=hashlib.sha256(EVALUATION_SOURCE.encode()).hexdigest())
def compatible_config(old, new):
    assert old == new, 'No-edges config mismatch; use a new output folder.'
for filename, value in [('config.json', CFG), ('identity.json', IDENTITY)]:
    p = RUN/filename
    if p.exists(): assert json.loads(p.read_text()) == value, 'Existing run metadata mismatch'
    else: write_json(value, p)
alpha, metrics = evaluate_validation()
assert alpha == 1.0
history = [dict(epoch=0, kgc_loss=0.0, alpha_clip=1.0,
                clip=metrics['1.0'], hybrid=metrics['1.0'])]
save_tensor(dict(model=model.state_dict(), cfg=CFG, identity=IDENTITY, epoch=0,
                 alpha_clip=1.0, metrics=metrics['1.0'], trained=False), CHECKPOINTS/'best_adamf.pt')
write_json(history, SEED_RUN/'history.json')
write_json(dict(epochs=0, best_mrr=metrics['1.0']['combined']['MRR'], seed=CFG['seed'],
                trained=False, policy='KG branch removed; CLIP-only control'), SEED_RUN/'complete.json')
val_q[['query_id','split','prediction_direction']].to_csv(VALIDATION_DIR/'validation_query_ids.csv', index=False)
(RUN/'evaluation_definitions.py').write_text(EVALUATION_SOURCE, encoding='utf-8')
print('NO EDGES: no KGC training; alpha_clip=1.0; random model weights are never used for ranking.')
