EVALUATOR_SOURCE = "def zscore(scores):\n    return (scores-scores.mean(-1,keepdim=True))/scores.std(-1,keepdim=True,correction=1).clamp_min(1e-6)\ndef rank_metrics(ranks):\n    ranks=np.asarray(ranks,dtype=np.int64)\n    if not len(ranks): return {'num_queries':0}\n    return {**{f'Recall@{k}':float(np.mean(ranks<=k)) for k in [1,5,10,50]},'MRR':float(np.mean(1.0/ranks)),'num_queries':len(ranks)}\ndef reranked_best_positive_rank(fused,positive_mask,baseline_rank):\n    best_positive=fused.masked_fill(~positive_mask,float('-inf')).max(1).values\n    inside=1+(fused>best_positive.unsqueeze(1)).sum(1)\n    return torch.where(positive_mask.any(1),inside,baseline_rank)\ndef evaluate_validation(model):\n    model.eval()\n    rank_lists={str(a):[] for a in CFG['alpha_clip_grid']}\n    with torch.no_grad():\n        for start in tqdm(range(0,len(val_q),CFG['query_batch_size']),desc='Validation'):\n            rows=val_q.iloc[start:start+CFG['query_batch_size']]\n            imgs=VAL_CACHE['top_rows'][start:start+len(rows)].to(DEVICE)\n            clip=VAL_CACHE['clip_scores'][start:start+len(rows)].to(DEVICE)\n            graph=model.score_images(rows,CLIP_QUERY['val'][start:start+len(rows)].to(DEVICE),imgs)\n            assert bool(torch.isfinite(graph).all())\n            clip_z,graph_z=zscore(clip),zscore(graph)\n            positives=torch.stack([torch.isin(imgs[i],torch.as_tensor(positive_rows['val'][row.query_id],device=DEVICE)) for i,row in enumerate(rows.itertuples(index=False))])\n            base=VAL_CACHE['baseline_rank'][start:start+len(rows)].to(DEVICE)\n            for alpha in CFG['alpha_clip_grid']:\n                fused=alpha*clip_z+(1-alpha)*graph_z\n                ranks=reranked_best_positive_rank(fused,positives,base)\n                if alpha==1.0: assert torch.equal(ranks,base), 'CLIP baseline mismatch'\n                rank_lists[str(alpha)].extend(ranks.cpu().tolist())\n    groups={'combined':np.ones(len(val_q),dtype=bool),'direct':val_q.prediction_direction.eq('direct').to_numpy(),'indirect-head':val_q.prediction_direction.eq('head').to_numpy(),'indirect-tail':val_q.prediction_direction.eq('tail').to_numpy()}\n    metrics={alpha:{group:rank_metrics(np.asarray(ranks)[mask]) for group,mask in groups.items()} for alpha,ranks in rank_lists.items()}\n    # Giống 04a: max theo MRR; tie lấy alpha đầu tiên trong grid.\n    best=max(CFG['alpha_clip_grid'],key=lambda a:metrics[str(a)]['combined']['MRR'])\n    return best,metrics\n"
def zscore(scores):
    return (scores-scores.mean(-1,keepdim=True))/scores.std(-1,keepdim=True,correction=1).clamp_min(1e-6)
def rank_metrics(ranks):
    ranks=np.asarray(ranks,dtype=np.int64)
    if not len(ranks): return {'num_queries':0}
    return {**{f'Recall@{k}':float(np.mean(ranks<=k)) for k in [1,5,10,50]},'MRR':float(np.mean(1.0/ranks)),'num_queries':len(ranks)}
def reranked_best_positive_rank(fused,positive_mask,baseline_rank):
    best_positive=fused.masked_fill(~positive_mask,float('-inf')).max(1).values
    inside=1+(fused>best_positive.unsqueeze(1)).sum(1)
    return torch.where(positive_mask.any(1),inside,baseline_rank)
def evaluate_validation(model):
    model.eval()
    rank_lists={str(a):[] for a in CFG['alpha_clip_grid']}
    with torch.no_grad():
        for start in tqdm(range(0,len(val_q),CFG['query_batch_size']),desc='Validation'):
            rows=val_q.iloc[start:start+CFG['query_batch_size']]
            imgs=VAL_CACHE['top_rows'][start:start+len(rows)].to(DEVICE)
            clip=VAL_CACHE['clip_scores'][start:start+len(rows)].to(DEVICE)
            graph=model.score_images(rows,CLIP_QUERY['val'][start:start+len(rows)].to(DEVICE),imgs)
            assert bool(torch.isfinite(graph).all())
            clip_z,graph_z=zscore(clip),zscore(graph)
            positives=torch.stack([torch.isin(imgs[i],torch.as_tensor(positive_rows['val'][row.query_id],device=DEVICE)) for i,row in enumerate(rows.itertuples(index=False))])
            base=VAL_CACHE['baseline_rank'][start:start+len(rows)].to(DEVICE)
            for alpha in CFG['alpha_clip_grid']:
                fused=alpha*clip_z+(1-alpha)*graph_z
                ranks=reranked_best_positive_rank(fused,positives,base)
                if alpha==1.0: assert torch.equal(ranks,base), 'CLIP baseline mismatch'
                rank_lists[str(alpha)].extend(ranks.cpu().tolist())
    groups={'combined':np.ones(len(val_q),dtype=bool),'direct':val_q.prediction_direction.eq('direct').to_numpy(),'indirect-head':val_q.prediction_direction.eq('head').to_numpy(),'indirect-tail':val_q.prediction_direction.eq('tail').to_numpy()}
    metrics={alpha:{group:rank_metrics(np.asarray(ranks)[mask]) for group,mask in groups.items()} for alpha,ranks in rank_lists.items()}
    # Giống 04a: max theo MRR; tie lấy alpha đầu tiên trong grid.
    best=max(CFG['alpha_clip_grid'],key=lambda a:metrics[str(a)]['combined']['MRR'])
    return best,metrics
