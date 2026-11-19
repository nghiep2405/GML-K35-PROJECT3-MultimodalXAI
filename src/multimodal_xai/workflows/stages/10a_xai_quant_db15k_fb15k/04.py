# Q0d. CLIP top-k và các hàm xếp hạng (y hệt notebook 07)
_TEXT_CACHE = {}
from multimodal_xai.workflows.runner import run_shared
run_shared('ranking', globals())





DATA_BY_KIND = {DATA[n].kind: DATA[n] for n in DATA}
