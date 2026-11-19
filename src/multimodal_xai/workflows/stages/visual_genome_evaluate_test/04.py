import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from tqdm.auto import tqdm

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

image_features = torch.load(
    FEATURE_ROOT / 'image_features.pt', map_location='cpu', weights_only=True
).float()
entity_features = torch.load(
    FEATURE_ROOT / 'entity_label_features.pt', map_location='cpu', weights_only=True
).float()
relation_features = torch.load(
    FEATURE_ROOT / 'relation_text_features.pt', map_location='cpu', weights_only=True
).float()
relation_mapping = pd.read_parquet(GRAPH_ROOT / 'relation_mapping.parquet')
relation_feature_index = pd.read_parquet(
    FEATURE_ROOT / 'relation_feature_index.parquet'
)
relation_join = relation_mapping.merge(
    relation_feature_index[['edge_family', 'relation', 'feature_row']],
    on=['edge_family', 'relation'], how='left', validate='one_to_one'
)
relation_type_features = torch.empty(
    (int(relation_mapping.relation_index.max()) + 1, 512), dtype=torch.float32
)
known = relation_join.feature_row.notna().to_numpy()
known_type = torch.from_numpy(
    relation_join.loc[known, 'relation_index'].to_numpy(np.int64)
)
known_row = torch.from_numpy(
    relation_join.loc[known, 'feature_row'].to_numpy(np.int64)
)
relation_type_features[known_type] = relation_features[known_row]
unk_feature_rows = relation_feature_index.loc[
    relation_feature_index.learnable_relation_index == 1, 'feature_row'
].to_numpy(np.int64)
unk_feature = F.normalize(
    relation_features[torch.from_numpy(unk_feature_rows)].mean(0), dim=0
)
missing_types = relation_join.loc[~known, 'relation_index'].to_numpy(np.int64)
assert set(missing_types).issubset({1})
if len(missing_types):
    relation_type_features[torch.from_numpy(missing_types)] = unk_feature
assert torch.isfinite(relation_type_features).all()

image_index = pd.read_parquet(
    FEATURE_ROOT / 'image_feature_index.parquet', columns=['image_id', 'feature_row']
)
image_to_component = dict(zip(image_index.image_id, image_index.feature_row))
global_to_local = torch.full((2_609_364,), -1, dtype=torch.long)
print('Feature tensors loaded')
