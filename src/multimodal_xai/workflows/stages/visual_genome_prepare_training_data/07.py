from multimodal_xai.visual_genome.features import prepare_text_embeddings
emb_manifest=prepare_text_embeddings(VG_ROOT,EXP_ROOT,shard_rows=16384,batch_size=512)
print(json.dumps({k:v for k,v in emb_manifest.items() if k!='parts'},indent=2))
