from multimodal_xai.visual_genome.pool import build_query_pool
query_manifest=build_query_pool(VG_ROOT,EXP_ROOT,per_image=20,seed=2026)
print(json.dumps(query_manifest,indent=2))
