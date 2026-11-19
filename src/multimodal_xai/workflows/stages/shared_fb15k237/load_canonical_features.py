DATA=REPO/'benchmarks/FB15K237-IMG'; DATA.mkdir(parents=True,exist_ok=True)
EMBED=REPO/'embeddings'; EMBED.mkdir(exist_ok=True)
adapter_names=['entity2id.txt','relation2id.txt','train2id.txt','valid2id.txt','test2id.txt','type_constrain.txt']
input_hashes={}
for name in adapter_names:
    source=REPRODUCTION/name; assert source.is_file(), f'Thiếu {source}'
    target=DATA/name
    shutil.copy2(source,target)
    expected=sha256(source); assert sha256(target)==expected
    input_hashes['adapter/'+name]=expected
staged={}
for key in ['entity_visual','entity_textual','entity_feature_index','modality_mask']:
    spec=BRIDGE['artifacts'][key]; source=resolve_data(spec['phase2_relative_path'])
    assert source.is_file() and source.stat().st_size==int(spec['size_bytes']), source
    assert sha256(source)==spec['sha256'], f'Canonical hash mismatch: {source}'
    target=EMBED/source.name
    shutil.copy2(source,target); assert sha256(target)==spec['sha256']
    staged[key]=target; input_hashes[key]=spec['sha256']
def mapping(path):
    with path.open(encoding='utf-8-sig') as f:
        n=int(f.readline()); rows=[s.rsplit(maxsplit=1) for s in f if s.strip()]
    assert len(rows)==n and all(len(r)==2 for r in rows)
    df=pd.DataFrame(rows,columns=['raw','id']); df['id']=df.id.astype('int64')
    assert df.raw.is_unique and np.array_equal(np.sort(df.id),np.arange(n))
    return df
entities=mapping(DATA/'entity2id.txt'); relations=mapping(DATA/'relation2id.txt')
N,R=len(entities),len(relations); assert (N,R)==(14541,237)
triples={}; triple_rows={}
for split,expected in [('train',272115),('valid',17535),('test',20466)]:
    with (DATA/f'{split}2id.txt').open() as f:
        declared=int(f.readline()); a=np.loadtxt(f,dtype=np.int64,ndmin=2)
    assert a.shape==(expected,3) and declared==expected
    assert a[:,:2].min()>=0 and a[:,:2].max()<N and a[:,2].min()>=0 and a[:,2].max()<R
    triples[split]=set(map(tuple,a.tolist())); assert len(triples[split])==expected
    triple_rows[split]=expected
for a,b in [('train','valid'),('train','test'),('valid','test')]: assert not triples[a]&triples[b], f'Overlap {a}/{b}'
train_entities={x for h,t,r in triples['train'] for x in [h,t]}
print('Entities không xuất hiện trong train:',N-len(train_entities))
idx=pd.read_parquet(staged['entity_feature_index'])
assert len(idx)==N and np.array_equal(np.sort(idx.entity_index),np.arange(N))
assert (idx.visual_row==idx.entity_index).all() and (idx.textual_row==idx.entity_index).all()
assert (idx.entity_id_raw.map(entities.set_index('raw').id)==idx.entity_index).all()
mask=pd.read_parquet(staged['modality_mask'])
assert len(mask)==N and np.array_equal(np.sort(mask.entity_index),np.arange(N))
mask=mask.set_index('entity_index').reindex(range(N))
assert mask[['has_visual','has_text']].notna().all().all()
assert mask[['has_visual','has_text']].isin([True,False,0,1]).all().all()
availability=torch.from_numpy(mask[['has_visual','has_text']].to_numpy(dtype=bool))
assert int((~availability[:,0]).sum())==280 and bool(availability[:,1].all())
for key,column in [('entity_visual',0),('entity_textual',1)]:
    z=torch.load(staged[key],map_location='cpu',weights_only=True)
    assert isinstance(z,torch.Tensor) and z.shape==(N,512) and z.is_floating_point()
    assert bool(torch.isfinite(z).all())
    assert torch.equal(z.norm(dim=1)==0,~availability[:,column])
    del z
mask_path=EMBED/'FB15K237-IMG-availability.pth'; torch.save(availability,mask_path)
# Identity dùng hash nội dung logical, không dùng hash serialization torch.save.
input_hashes['availability_logical']=hashlib.sha256(availability.numpy().tobytes()).hexdigest()
PROVENANCE=dict(commit=COMMIT,source_mode=SOURCE_MODE,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),entities=N,relations=R,triples=triple_rows,input_sha256=input_hashes,bridge_sha256=sha256(MANIFEST_PATH),dataset='fb15k237_img_v1',checkpoint_selection='best_validation_retrieval_mrr',feature_policy='frozen_entity_features_trainable_projections',modality_policy='mask_before_attention_softmax',kg_score_for_retrieval='forward_score_higher_is_better; predict_is_negative_forward',train_isolated_entities=N-len(train_entities))
del triples
print('PREFLIGHT PASS: official split, features và modality mask.')
