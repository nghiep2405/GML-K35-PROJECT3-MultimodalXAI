from multimodal_xai.visual_genome.negatives import build_hard_negatives
hard_manifest=build_hard_negatives(VG_ROOT,EXP_ROOT,batch_size=128,k=50)
print(json.dumps({k:v for k,v in hard_manifest.items() if k!='parts'},indent=2))
