from huggingface_hub import model_info
from transformers import CLIPModel, CLIPTokenizerFast
text_meta_path=CACHE/'text_encoder.json'
if text_meta_path.is_file():
    TEXT_META=json.loads(text_meta_path.read_text())
else:
    model_name=clip_metadata['model_name']
    revisions=[x for x in clip_metadata.get('resolved_commits',[]) if x and str(x).lower() not in ['none','null']]
    revision=revisions[0] if revisions else model_info(model_name,revision=clip_metadata.get('requested_revision','main')).sha
    TEXT_META=dict(model_name=model_name,revision=revision,image_encoder_revision_pinned=bool(revisions),source_metadata_sha256=BRIDGE['artifacts']['clip_feature_metadata']['sha256'])
    write_json(TEXT_META,text_meta_path)
assert TEXT_META['model_name']==clip_metadata['model_name']
print('Text encoder:',TEXT_META)
if not TEXT_META['image_encoder_revision_pinned']:
    print('Lưu ý: metadata ảnh gốc không có commit; không thể chứng minh encoder revision trùng tuyệt đối. Dùng cache này chung cho các model.')
CLIP_QUERY={}
encoder=None; tokenizer=None
for split,df in [('train',train_q),('val',val_q)]:
    path=CACHE/f'{split}_clip_text.pt'
    identity=hashlib.sha256((json.dumps(TEXT_META,sort_keys=True)+df[['query_id','query_text']].to_json(orient='records',force_ascii=False)).encode()).hexdigest()
    if path.is_file():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['identity']==identity and payload['features'].shape==(len(df),512), 'Text cache mismatch'
        CLIP_QUERY[split]=payload['features']; continue
    if encoder is None:
        tokenizer=CLIPTokenizerFast.from_pretrained(TEXT_META['model_name'],revision=TEXT_META['revision'])
        encoder=CLIPModel.from_pretrained(TEXT_META['model_name'],revision=TEXT_META['revision']).to(DEVICE).eval()
        encoder.requires_grad_(False)
    texts=df.query_text.tolist(); unique=list(dict.fromkeys(texts)); mapping_text={t:i for i,t in enumerate(unique)}
    outputs=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(unique),256),desc=f'Encode {split} queries'):
            inputs=tokenizer(unique[start:start+256],padding=True,truncation=True,max_length=77,return_tensors='pt').to(DEVICE)
            # Explicit text encoder + projection avoids get_text_features return-type changes.
            output=encoder.text_projection(encoder.text_model(**inputs,return_dict=True).pooler_output)
            outputs.append(F.normalize(output.float(),dim=1).cpu())
    unique_z=torch.cat(outputs)
    features=unique_z[torch.tensor([mapping_text[t] for t in texts])]
    assert features.shape==(len(df),512) and bool(torch.isfinite(features).all())
    atomic_torch_save(dict(identity=identity,features=features),path)
    CLIP_QUERY[split]=features
del encoder,tokenizer
torch.cuda.empty_cache()
