# Xem nhanh một hình
from IPython.display import Image as _IPImage, display
display(_IPImage(filename=str(OUT / "cases" / CASE_RESULTS[0]["figure"])))
