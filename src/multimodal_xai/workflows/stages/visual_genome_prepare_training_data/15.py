from multimodal_xai.visual_genome.audit import audit_pool, audit_sampling
pool_report=audit_pool(VG_ROOT,EXP_ROOT)
sampling_report=audit_sampling(EXP_ROOT,epochs=8,seed=2026)
GUARD.save_json(EXP_ROOT/'reports/preparation_audit.json',{'pool':pool_report,'sampling':sampling_report})
print(json.dumps({'pool':pool_report,'sampling':sampling_report},indent=2))
