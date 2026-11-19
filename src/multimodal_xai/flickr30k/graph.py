"""Spatial relations and deterministic graph ablations."""
import math

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

EDGE_DIM = 8


def relation_type(a: list[float], b: list[float]) -> tuple[int, int] | None:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    dx, dy = bcx - acx, bcy - acy
    if intersection > 0:
        return 3, 3
    if math.hypot(dx, dy) >= 0.35:
        return None
    if abs(dx) >= abs(dy):
        return (4, 5) if dx >= 0 else (5, 4)
    return (6, 7) if dy >= 0 else (7, 6)


def graph_for(nodes: torch.Tensor, metadata: dict[str, object], image_index: int, *, edge_mode: str, seed: int) -> Data:
    count = len(nodes)
    edges, types = [], []
    for node in range(count):
        edges.append((node, node))
        types.append(0)
    if edge_mode != "no_edge":
        for node in range(1, count):
            edges.extend(((0, node), (node, 0)))
            types.extend((1, 2))
        boxes = [region["box_normalized"] for region in metadata["regions"]]
        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                pair = relation_type(boxes[left], boxes[right])
                if pair is not None:
                    edges.extend(((left + 1, right + 1), (right + 1, left + 1)))
                    types.extend(pair)
    if edge_mode == "random_edges" and len(edges) > count:
        generator = torch.Generator().manual_seed(seed * 1_000_003 + image_index)
        for index in range(count, len(edges)):
            edges[index] = (
                int(torch.randint(count, (1,), generator=generator)),
                int(torch.randint(count, (1,), generator=generator)),
            )
    edge_index = torch.tensor(edges, dtype=torch.int64).T.contiguous()
    edge_attr = F.one_hot(torch.tensor(types), num_classes=EDGE_DIM).float()
    return Data(x=nodes.float(), edge_index=edge_index, edge_attr=edge_attr)
