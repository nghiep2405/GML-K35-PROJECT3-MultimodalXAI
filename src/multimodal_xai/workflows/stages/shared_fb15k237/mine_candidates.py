def stable_descending(scores):
    return torch.argsort(scores,dim=-1,descending=True,stable=True)
def stable_topk(scores,k):
    # Exact stable top-k without sorting the entire candidate pool.
    threshold=torch.topk(scores,k).values[-1]
    eligible=(scores>=threshold).nonzero(as_tuple=True)[0]
    return eligible[stable_descending(scores[eligible])[:k]]
def build_clip_cache(split,df):
    path=CACHE/f'{split}_clip_candidates.pt'
    text_hash=sha256(CACHE/f'{split}_clip_text.pt')
    identity=hashlib.sha256((cache_key+text_hash).encode()).hexdigest()
    if path.is_file():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['identity']==identity
        return payload
    pool_df=candidates.loc[candidates['split']==split]
    pool=np.sort(pool_df.image_id.map(image_rows).to_numpy(np.int64))
    assert len(np.unique(pool))==len(pool)
    features=F.normalize(torch.from_numpy(np.array(image_array[pool],copy=True)).float(),dim=1).to(DEVICE)
    positions={int(r):i for i,r in enumerate(pool)}
    k=min(CFG['hard_negative_pool'] if split=='train' else CFG['rerank_topk'],len(pool))
    assert k>=CFG['train_negatives'] if split=='train' else k>=50
    hard=[]; top_rows=[]; top_scores=[]; ranks=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(df),64),desc=f'CLIP {split} candidates'):
            score=CLIP_QUERY[split][start:start+64].to(DEVICE)@features.T
            for j in range(len(score)):
                row=df.iloc[start+j]; positives=positive_rows[split][row.query_id]
                assert all(int(r) in positions for r in positives)
                pos=torch.tensor([positions[int(r)] for r in positives],device=DEVICE)
                if split=='train':
                    score[j,pos]=float('-inf')
                    order=stable_topk(score[j],k)
                    selected=pool[order.cpu().numpy()]
                    selected=selected[~np.isin(selected,positives)]
                    assert len(selected)>=CFG['train_negatives'], 'Không đủ negatives'
                    hard.append(torch.tensor(selected[:k],dtype=torch.long))
                else:
                    # Giống 04a: rank = 1 + số score lớn hơn best-positive score.
                    best_score=score[j,pos].max()
                    before=score[j]>best_score
                    positive_position=int(before.sum())
                    chosen=stable_topk(score[j],k)
                    top_rows.append(torch.from_numpy(pool[chosen.cpu().numpy()].copy()))
                    top_scores.append(score[j,chosen].cpu())
                    ranks.append(1+positive_position)
    if split=='train':
        payload=dict(identity=identity,negative_rows=hard)
    else:
        payload=dict(identity=identity,top_rows=torch.stack(top_rows),clip_scores=torch.stack(top_scores),baseline_rank=torch.tensor(ranks))
    atomic_torch_save(payload,path)
    del features; torch.cuda.empty_cache()
    return payload
TRAIN_CACHE=build_clip_cache('train',train_q)
VAL_CACHE=build_clip_cache('val',val_q)
print('Hard-negative queries:',len(TRAIN_CACHE['negative_rows']))
print('Validation top-k:',VAL_CACHE['top_rows'].shape)
