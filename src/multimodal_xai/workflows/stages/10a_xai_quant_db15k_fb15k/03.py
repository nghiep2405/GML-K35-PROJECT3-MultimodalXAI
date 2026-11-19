# Q0c. Mô hình và bộ chấm điểm (y hệt notebook 07)
# C0c. Mô hình: dựng lại đúng lớp của nhóm (AdvMixRotatE từ AdaMF-MAT, bản nhúng trong notebook 04b1) và nạp checkpoint
from multimodal_xai.workflows.runner import run_shared
run_shared('models', globals())








SCORERS = {(name, s): Scorer(DATA[name], s) for name in DATASET_NAMES for s in SEEDS}
for (name, s), sc in SCORERS.items():
    print(f"{name} seed {s}: epoch {sc.epoch}, α_clip = {sc.alpha}, dim_e = {sc.dim_e}")
