# Final test: locked validation checkpoint and alpha; no training or tuning.
from datetime import datetime, timezone
from transformers import CLIPModel, CLIPTokenizerFast

for seed in SEEDS:
    assert (RUN/f'seed_{seed}'/'complete.json').is_file(), 'Finish training / early stopping before final test.'
    assert (RUN/f'seed_{seed}'/'best_adamf.pt').is_file(), 'Missing best validation checkpoint.'
# Verify original artifacts before evaluating held-out data.
for key in ['queries','relevance','candidate_pools','entity_image','image_feature_index','clip_image_features']:
    spec=BRIDGE['artifacts'][key]
    assert sha256(resolve_data(spec['phase2_relative_path']))==spec['sha256'], f'Input changed: {key}'
test_q=queries.loc[queries['split'].eq('test')].sort_values('query_id').reset_index(drop=True)
assert len(test_q)>0 and test_q.query_id.is_unique
assert test_q.is_eval_only.all()
assert test_q.prediction_direction.isin(['direct','head','tail']).all()
assert not test_q.query_id.isin(queries.loc[queries['split'].isin(['train','val']),'query_id']).any()
test_images=candidates.loc[candidates['split'].eq('test'),'image_id']
assert len(test_images)>0 and test_images.is_unique
assert not test_images.isin(candidates.loc[candidates['split'].isin(['train','val']),'image_id']).any()
assert test_images.isin(image_rows.index).all()
test_rel=relevance.loc[relevance.query_id.isin(test_q.query_id)].copy()
assert test_rel.image_id.isin(test_images).all()
test_rel['row']=test_rel.image_id.map(image_rows)
assert test_rel.row.notna().all()
positive_rows['test']=test_rel.groupby('query_id').row.apply(lambda x:np.unique(x.to_numpy(np.int64))).to_dict()
assert set(positive_rows['test'])==set(test_q.query_id)

# Pinned encoder and identical query encoding / candidate ranking as validation.
text_path=CACHE/'test_clip_text.pt'
text_identity=hashlib.sha256((json.dumps(TEXT_META,sort_keys=True)+test_q[['query_id','query_text']].to_json(orient='records',force_ascii=False)).encode()).hexdigest()
if text_path.is_file():
    payload=torch.load(text_path,map_location='cpu',weights_only=True)
    assert payload['identity']==text_identity and payload['features'].shape==(len(test_q),512)
    CLIP_QUERY['test']=payload['features']
else:
    tokenizer_test=CLIPTokenizerFast.from_pretrained(TEXT_META['model_name'],revision=TEXT_META['revision'])
    encoder_test=CLIPModel.from_pretrained(TEXT_META['model_name'],revision=TEXT_META['revision']).to(DEVICE).eval()
    encoder_test.requires_grad_(False)
    texts=test_q.query_text.tolist(); unique=list(dict.fromkeys(texts)); text_index={t:i for i,t in enumerate(unique)}
    encoded=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(unique),256),desc='Encode test queries'):
            tokens=tokenizer_test(unique[start:start+256],padding=True,truncation=True,max_length=77,return_tensors='pt').to(DEVICE)
            features=encoder_test.text_projection(encoder_test.text_model(**tokens,return_dict=True).pooler_output)
            encoded.append(F.normalize(features.float(),dim=1).cpu())
    CLIP_QUERY['test']=torch.cat(encoded)[torch.tensor([text_index[t] for t in texts])]
    assert CLIP_QUERY['test'].shape==(len(test_q),512) and torch.isfinite(CLIP_QUERY['test']).all()
    atomic_torch_save(dict(identity=text_identity,features=CLIP_QUERY['test']),text_path)
    del encoder_test,tokenizer_test,encoded,tokens,features
    torch.cuda.empty_cache()
TEST_CACHE=build_clip_cache('test',test_q)

for seed in SEEDS:
    folder=RUN/f'seed_{seed}'
    best_path=folder/'best_adamf.pt'
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    assert saved['identity']==IDENTITY and saved['cfg']==CFG and saved['seed']==seed
    alpha=float(saved['alpha_clip'])
    assert 0.<=alpha<=1.
    test_model=make_model(seed)
    test_model.load_state_dict(saved['model'])
    test_model.eval().requires_grad_(False)
    records=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(test_q),CFG['query_batch_size']),desc=f'Test seed {seed} (locked alpha)'):
            rows=test_q.iloc[start:start+CFG['query_batch_size']]
            imgs=TEST_CACHE['top_rows'][start:start+len(rows)].to(DEVICE)
            clip=TEST_CACHE['clip_scores'][start:start+len(rows)].to(DEVICE)
            graph=test_model.score_images(rows,CLIP_QUERY['test'][start:start+len(rows)].to(DEVICE),imgs)
            assert torch.isfinite(graph).all()
            # Match FB15K-237 validation: scorer handles direct AND indirect queries.
            fused=alpha*zscore(clip)+(1.-alpha)*zscore(graph)
            positives=torch.stack([torch.isin(imgs[i],torch.as_tensor(positive_rows['test'][row.query_id],device=DEVICE)) for i,row in enumerate(rows.itertuples(index=False))])
            baseline=TEST_CACHE['baseline_rank'][start:start+len(rows)].to(DEVICE)
            hybrid=reranked_best_positive_rank(fused,positives,baseline)
            if alpha==1.: assert torch.equal(hybrid,baseline), 'CLIP baseline mismatch'
            for i,row in enumerate(rows.itertuples(index=False)):
                records.append(dict(query_id=row.query_id,direction=row.prediction_direction,
                    clip_rank=int(baseline[i]),hybrid_rank=int(hybrid[i])))
    ranks=pd.DataFrame(records)
    masks={'combined':np.ones(len(ranks),dtype=bool),'direct':ranks.direction.eq('direct').to_numpy(),
           'indirect':ranks.direction.ne('direct').to_numpy(),
           'indirect-head':ranks.direction.eq('head').to_numpy(),'indirect-tail':ranks.direction.eq('tail').to_numpy()}
    result=[]
    for method,column in [('clip_only','clip_rank'),('clip_plus_graph','hybrid_rank')]:
        for group,mask in masks.items():
            result.append(dict(method=method,group=group,seed=seed,epoch=saved['epoch'],
                alpha_clip=1. if method=='clip_only' else alpha,**rank_metrics(ranks.loc[mask,column].to_numpy())))
    frame=pd.DataFrame(result)
    out=folder/('test_locked_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    frame.to_csv(out/'test_metrics.csv',index=False)
    ranks.to_csv(out/'test_query_ranks.csv',index=False)
    write_json(dict(checkpoint=str(best_path),checkpoint_sha256=sha256(best_path),
        selected_validation_epoch=saved['epoch'],locked_alpha_clip=alpha,seed=seed,
        rerank_topk=CFG['rerank_topk'],encoder=TEXT_META,identity=IDENTITY,
        num_queries=len(test_q),num_candidates=len(test_images),
        protocol='Same scorer, normalization, best-positive optimistic rank, stable CLIP top-k and outside-top-k fallback as validation; no test tuning.'),out/'test_protocol.json')
    display(frame)
    print('TEST outputs:',out)
    del test_model,saved
    torch.cuda.empty_cache()
