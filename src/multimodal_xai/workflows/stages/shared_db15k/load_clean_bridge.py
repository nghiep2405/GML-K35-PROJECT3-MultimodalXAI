interface=json.loads((CLEAN/'training_interface.json').read_text())
assert interface['protocol_id']=='db15k_clean_from_train_v1'
assert interface['retrieval_protocol_id']=='db15k_clean_retrieval_v2'
manifest=json.loads((RETRIEVAL/'manifest.json').read_text())
ref_path=verified('canonical_references',manifest['artifacts']['canonical_references'])
references=json.loads(ref_path.read_text())
feature_refs=json.loads((CLEAN/'feature_references.json').read_text())
paths={}
for key in ['queries','relevance','candidate_pools']:
    paths[key]=verified(key,manifest['artifacts'][key])
    assert paths[key].is_relative_to(RETRIEVAL.resolve())
for key in ['entity_image','image_feature_index','clip_image_features','clip_feature_metadata']:
    paths[key]=verified(key,references['artifacts'][key])
paths['textual']=verified('textual',feature_refs['textual'])
for name in ['training_interface.json','feature_references.json']:
    HASHES[name]=sha(CLEAN/name)
def read_mapping(path):
    with path.open(encoding='utf-8-sig') as f:
        count=int(f.readline());rows=[x.rsplit(maxsplit=1) for x in f if x.strip()]
    frame=pd.DataFrame(rows,columns=['raw','id']);frame['id']=frame.id.astype('int64')
    assert len(frame)==count and frame.raw.is_unique and np.array_equal(np.sort(frame.id),np.arange(count))
    HASHES[path.name]=sha(path);return frame
kgroot=CLEAN/'reproduction'
entities=read_mapping(kgroot/'entity2id.txt');relations=read_mapping(kgroot/'relation2id.txt')
N,R=len(entities),len(relations);assert (N,R)==(12842,279)
splits={}
for split,count in [('train',71300),('valid',7922)]:
    p=kgroot/f'{split}2id.txt'
    with p.open() as f:declared=int(f.readline());a=np.loadtxt(f,dtype=np.int64,ndmin=2)
    assert declared==count and a.shape==(count,3) and len(np.unique(a,axis=0))==count
    assert a[:,:2].min()>=0 and a[:,:2].max()<N and a[:,2].min()>=0 and a[:,2].max()<R
    splits[split]=a;HASHES[split]=sha(p)
train_set=set(map(tuple,splits['train'].tolist()));valid_set=set(map(tuple,splits['valid'].tolist()))
assert not train_set&valid_set
train_entities=set(splits['train'][:,:2].ravel().tolist())
queries=pd.read_parquet(paths['queries']);relevance=pd.read_parquet(paths['relevance']);pools=pd.read_parquet(paths['candidate_pools'])
# Drop test rows immediately; no test labels or candidate features used for fitting/tuning.
queries=queries.loc[queries['split'].isin(['train','val'])].copy()
relevance=relevance.loc[relevance.query_id.isin(queries.query_id)].copy()
pools=pools.loc[pools['split'].isin(['train','val'])].copy()
assert queries.query_id.is_unique and pools.image_id.is_unique
assert queries.prediction_direction.isin(['direct','head','tail']).all()
assert not queries.loc[queries['split']=='train','is_eval_only'].any()
assert queries.loc[queries['split']=='val','is_eval_only'].all()
assert queries.groupby('split').size().to_dict()=={'train':130988,'val':5023}
for row in queries.loc[queries.prediction_direction!='direct'].itertuples(index=False):
    a,t,r=int(row.anchor_entity_index),int(row.target_entity_index),int(row.relation_index)
    triple=(a,t,r) if row.prediction_direction=='tail' else (t,a,r)
    assert triple in (train_set if row.split=='train' else valid_set),row.query_id
idx=pd.read_parquet(paths['image_feature_index']);links=pd.read_parquet(paths['entity_image'])
assert idx.image_id.is_unique and np.array_equal(np.sort(idx.feature_row),np.arange(len(idx)))
assert links.image_id.is_unique,'This DB15K bridge expects one entity per image'
assert links.entity_index.between(0,N-1).all()
row_map=idx.set_index('image_id').feature_row
assert pools.image_id.isin(row_map.index).all() and pools.image_id.isin(links.image_id).all()
image_entities=idx.sort_values('feature_row').image_id.map(links.set_index('image_id').entity_index)
assert image_entities.notna().all()
IMAGE_ENTITIES=torch.tensor(image_entities.to_numpy(np.int64))
image_array=np.load(paths['clip_image_features'],mmap_mode='r',allow_pickle=False)
assert image_array.shape==(25925,512)
textual=torch.load(paths['textual'],map_location='cpu',weights_only=True).float()
assert textual.shape==(N,768) and bool(torch.isfinite(textual).all())
assert bool((textual.norm(dim=1)>0).all())
# Use TRAIN images only to construct visual inputs to the learned projector.
train_rows=pools.loc[pools['split']=='train','image_id'].map(row_map).to_numpy(np.int64)
train_visual=torch.zeros(N,512);counts=torch.zeros(N)
for start in range(0,len(train_rows),4096):
    rr=train_rows[start:start+4096];features=torch.tensor(np.array(image_array[rr],copy=True))
    assert bool(torch.isfinite(features).all())
    ids=IMAGE_ENTITIES[rr];train_visual.index_add_(0,ids,features);counts.index_add_(0,ids,torch.ones(len(rr)))
train_visual=F.normalize(train_visual,dim=1)
availability=torch.stack([counts>0,torch.ones(N,dtype=torch.bool)],dim=1)
val_q=queries.loc[queries['split']=='val'].sort_values('query_id').reset_index(drop=True)
val_rel=relevance.loc[relevance.query_id.isin(val_q.query_id)]
assert set(val_rel.query_id)==set(val_q.query_id)
val_pool=np.sort(pools.loc[pools['split']=='val','image_id'].map(row_map).to_numpy(np.int64))
positive_rows=val_rel.assign(row=val_rel.image_id.map(row_map)).groupby('query_id').row.apply(lambda x:np.unique(x.to_numpy(np.int64))).to_dict()
assert all(np.isin(rows,val_pool).all() for rows in positive_rows.values())
print('PREFLIGHT PASS; train visual coverage:',int((counts>0).sum()),'/',N)
print('Train KGC / valid KGC / valid retrieval:',len(train_set),len(valid_set),len(val_q))
