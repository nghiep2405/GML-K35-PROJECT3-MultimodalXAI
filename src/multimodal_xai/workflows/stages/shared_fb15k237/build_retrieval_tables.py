import random, math
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
DEVICE=torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32=False
def atomic_torch_save(payload,path):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def write_json(payload,path):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)
def verified_artifact(key):
    spec=BRIDGE['artifacts'][key]; source=resolve_data(spec['phase2_relative_path'])
    assert sha256(source)==spec['sha256'], f'Hash mismatch: {key}'
    target=LOCAL_CACHE/source.name
    if not target.is_file() or sha256(target)!=spec['sha256']:
        shutil.copy2(source,target)
    assert sha256(target)==spec['sha256']
    return target
queries=pd.read_parquet(verified_artifact('queries'))
relevance=pd.read_parquet(verified_artifact('relevance'))
candidates=pd.read_parquet(verified_artifact('candidate_pools'))
links=pd.read_parquet(verified_artifact('entity_image'))
image_index=pd.read_parquet(verified_artifact('image_feature_index'))
image_array=np.load(verified_artifact('clip_image_features'),mmap_mode='r',allow_pickle=False)
clip_metadata=json.loads(verified_artifact('clip_feature_metadata').read_text())
assert image_array.shape==(139814,512)
assert queries.query_id.is_unique and image_index.image_id.is_unique
assert np.array_equal(np.sort(image_index.feature_row),np.arange(len(image_index)))
image_rows=pd.Series(image_index.feature_row.to_numpy(),index=image_index.image_id)
split_by_image=candidates.set_index('image_id')['split']
assert split_by_image.index.is_unique
links=links[['image_id','entity_index']].drop_duplicates().copy()
links['feature_row']=links.image_id.map(image_rows)
assert links.feature_row.notna().all() and links.entity_index.between(0,N-1).all()
links['feature_row']=links.feature_row.astype('int64')
links['split']=links.image_id.map(split_by_image)
assert links['split'].notna().all()
image_entity_lists=links.groupby('feature_row').entity_index.apply(list).to_dict()
max_links=max(map(len,image_entity_lists.values()))
IMAGE_ENTITY=torch.full((len(image_index),max_links),-1,dtype=torch.long)
for row,ids in image_entity_lists.items(): IMAGE_ENTITY[row,:len(ids)]=torch.tensor(ids)
assert bool((IMAGE_ENTITY>=0).any(1).all())
PROTOCOL=dict(dataset='fb15k237_img_v1',query_version='fb15k237_query_v1',
    source_hashes={k:BRIDGE['artifacts'][k]['sha256'] for k in ['queries','relevance','candidate_pools','entity_image','image_feature_index','clip_image_features','entity_textual']},
    max_train_queries=CFG['max_train_queries'],selection_seed=CFG['selection_seed'],
    train_negatives=CFG['train_negatives'],hard_negative_pool=CFG['hard_negative_pool'],
    rerank_topk=CFG['rerank_topk'],validation_queries=CFG['validation_queries'],
    alpha_clip_grid=CFG['alpha_clip_grid'],normalization=CFG['normalization'],
    ranking='rerank_CLIP_topK_keep_outside_order',ties='optimistic_rank_strictly_greater_like_04a',topk_boundary_ties='stable_candidate_feature_row',recall_definition='query_hit_rate_at_K_using_best_relevant_rank',
    positive_policy='all_target_images_in_query_split',visual_pool='train_images_only_v2')
cache_key=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()[:20]
CACHE=CACHE_ROOT/cache_key; CACHE.mkdir(parents=True,exist_ok=True)
write_json(PROTOCOL,CACHE/'protocol.json')
train_links=links.loc[links['split']=='train']
train_visual=torch.zeros(N,512); train_count=torch.zeros(N)
for start in range(0,len(train_links),8192):
    chunk=train_links.iloc[start:start+8192]
    ids=torch.from_numpy(chunk.entity_index.to_numpy(np.int64))
    feats=torch.from_numpy(np.array(image_array[chunk.feature_row.to_numpy()],copy=True))
    train_visual.index_add_(0,ids,feats)
    train_count.index_add_(0,ids,torch.ones(len(chunk)))
train_visual=F.normalize(train_visual,dim=1)
train_availability=availability.clone(); train_availability[:,0]=train_count>0
assert bool((train_visual.norm(dim=1)==0).eq(~train_availability[:,0]).all())
PROVENANCE.update(training_task='query_image_retrieval_cross_entropy',
    adaptation='AdaMF_attention_RotatE_for_indirect_queries_text_projection_for_direct',
    visual_training_policy='repool_entity_visuals_from_train_images_only',
    max_epochs=CFG['epochs'],checkpoint_selection='best_validation_retrieval_mrr')
TEXTUAL=torch.load(staged['entity_textual'],map_location='cpu',weights_only=True).float()
print('Train visual coverage:',int(train_availability[:,0].sum()),'/',N)
print('Cache:',CACHE,'; max entities/image:',max_links)
# Positives được lấy từ relevance, không dùng ảnh test cho train/val.
positive_rows={}
for split in ['train','val']:
    qids=set(queries.loc[queries['split']==split,'query_id'])
    part=relevance.loc[relevance.query_id.isin(qids),['query_id','image_id']].copy()
    part['row']=part.image_id.map(image_rows)
    assert part.row.notna().all()
    positive_rows[split]=part.groupby('query_id').row.apply(lambda x:np.unique(x.to_numpy(np.int64))).to_dict()
    assert qids==set(positive_rows[split])
train_q=queries.loc[queries['split']=='train'].copy().sort_values('query_id').reset_index(drop=True)
train_q=train_q.iloc[np.sort(np.random.default_rng(CFG['selection_seed']).choice(len(train_q),min(len(train_q),CFG['max_train_queries']),replace=False))].reset_index(drop=True)
val_q=queries.loc[queries['split']=='val'].copy().sort_values('query_id').reset_index(drop=True)
if CFG['validation_queries'] is not None and CFG['validation_queries']<len(val_q):
    val_q=val_q.iloc[np.sort(np.random.default_rng(CFG['selection_seed']).choice(len(val_q),CFG['validation_queries'],replace=False))].reset_index(drop=True)
assert not train_q.is_eval_only.any() and val_q.is_eval_only.all()
for name,df in [('train',train_q),('val',val_q)]:
    df.to_parquet(CACHE/f'{name}_queries.parquet',index=False)
print('Selected train/val queries:',len(train_q),len(val_q))
