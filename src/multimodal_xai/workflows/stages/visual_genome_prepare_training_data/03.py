from multimodal_xai.visual_genome.guard import verify_frozen_val_test
# Full SHA-256 compares original val/test protocol parquet files with frozen official hashes.
print(json.dumps(verify_frozen_val_test(VG_ROOT,full_sha=True),indent=2))
