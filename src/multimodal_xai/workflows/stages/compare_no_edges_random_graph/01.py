from pathlib import Path

# Nếu CSV ở Google Drive, để True. Nếu upload CSV từ Molab, đặt False.
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

ROOT = Path('/content/drive/MyDrive/MLGs - Project/dataset/adamf/model/adamf_v1')
RUNS = {
    'no_edges': 'db15k_clean_kgc_to_retrieval_seed2027_ablation_no_edges_v1',
    'random_graph': 'db15k_clean_kgc_to_retrieval_seed2027_ablation_random_graph_v1',
}
# Để tự tìm lần test gần nhất trong từng thư mục trên, giữ None.
# Nếu upload thủ công vào Colab, đổi tên CSV như dưới và bỏ dấu # ở hai dòng.
CSV_PATHS = {
    'no_edges': None,       # '/content/no_edges_test_metrics.csv'
    'random_graph': None,   # '/content/random_graph_test_metrics.csv'
}
