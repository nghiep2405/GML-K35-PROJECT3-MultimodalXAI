from transformers import CLIPModel,CLIPTokenizerFast
from huggingface_hub import model_info
clip_meta=json.loads(paths['clip_feature_metadata'].read_text())
encoder_name=clip_meta['encoder_name']
meta_path=RUN/'clip_encoder.json'
if meta_path.exists():encoder_meta=json.loads(meta_path.read_text())
else:
    revision=model_info(encoder_name,revision=clip_meta.get('checkpoint_revision','main')).sha
    encoder_meta=dict(name=encoder_name,revision=revision,image_metadata_sha256=HASHES['clip_feature_metadata'],image_revision_exactly_known=False)
    write_json(encoder_meta,meta_path)
assert encoder_meta['name']==encoder_name and encoder_meta['image_metadata_sha256']==HASHES['clip_feature_metadata']
cache_identity=hashlib.sha256(json.dumps(dict(hashes=HASHES,encoder=encoder_meta,topk=CFG['topk'],version='clip-val-v1'),sort_keys=True).encode()).hexdigest()
cache_path=CACHE_DIR/'validation_clip_cache.pt'
if cache_path.exists():
    CLIP_CACHE=torch.load(cache_path,map_location='cpu',weights_only=True)
    assert CLIP_CACHE['identity']==cache_identity
else:
    tokenizer=CLIPTokenizerFast.from_pretrained(encoder_name,revision=encoder_meta['revision'])
    encoder=CLIPModel.from_pretrained(encoder_name,revision=encoder_meta['revision']).to(DEVICE).eval()
    encoder.requires_grad_(False);output=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(val_q),128),desc='CLIP validation text'):
            tokens=tokenizer(val_q.query_text.iloc[start:start+128].tolist(),padding=True,truncation=True,max_length=77,return_tensors='pt').to(DEVICE)
            out=encoder.text_projection(encoder.text_model(**tokens,return_dict=True).pooler_output)
            output.append(F.normalize(out.float(),dim=1).cpu())
    query_z=torch.cat(output);del encoder,tokenizer;torch.cuda.empty_cache()
    pool_z=F.normalize(torch.tensor(np.array(image_array[val_pool],copy=True)).float(),dim=1).to(DEVICE)
    positions={int(row):i for i,row in enumerate(val_pool)}
    top_rows=[];top_scores=[];base_rank=[];positive_mask=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(val_q),64),desc='CLIP validation rank'):
            scores=query_z[start:start+64].to(DEVICE)@pool_z.T
            for j in range(len(scores)):
                positives=positive_rows[val_q.iloc[start+j].query_id]
                pos=torch.tensor([positions[int(x)] for x in positives],device=DEVICE)
                score=scores[j];base_rank.append(int((score>score[pos].max()).sum())+1)
                order=torch.argsort(score,descending=True,stable=True)[:min(CFG['topk'],len(val_pool))]
                rows=torch.tensor(val_pool[order.cpu().numpy()])
                top_rows.append(rows);top_scores.append(score[order].cpu())
                positive_mask.append(torch.isin(rows,torch.tensor(positives)))
    CLIP_CACHE=dict(identity=cache_identity,top_rows=torch.stack(top_rows),scores=torch.stack(top_scores),
        baseline_rank=torch.tensor(base_rank),positive_mask=torch.stack(positive_mask))
    save_tensor(CLIP_CACHE,cache_path);del pool_z;torch.cuda.empty_cache()
print('CLIP validation MRR:',float((1./CLIP_CACHE['baseline_rank'].float()).mean()))
print('Encoder pin:',encoder_meta)
