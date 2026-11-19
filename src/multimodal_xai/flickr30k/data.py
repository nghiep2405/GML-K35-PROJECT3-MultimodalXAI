"""Build audited Flickr30k retrieval tensors and train hard negatives."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


from ..artifacts.atomic import atomic_save


PROJECT_ROOT = Path(os.environ.get("GRAPHGAT_PROJECT_ROOT", Path.cwd()))
PROCESSED = PROJECT_ROOT / "data/Flickr30k/processed/clip_vit_b32_openai"
OUTPUT = Path(os.environ.get("FLICKR_NEGATIVES_ROOT", PROJECT_ROOT / "data/Flickr30k/retrieval"))
TOP_K = int(os.environ.get("FLICKR_HARD_NEGATIVE_K", "50"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split: str) -> dict[str, object]:
    image_ids, image_embeddings, graph_nodes, graph_metadata = [], [], [], []
    for path in sorted((PROCESSED / "images" / split).glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        offsets = shard["node_offsets"].tolist()
        for index, image_id in enumerate(shard["image_ids"]):
            nodes = shard["node_embeddings"][offsets[index] : offsets[index + 1]]
            image_ids.append(image_id)
            image_embeddings.append(nodes[0])
            graph_nodes.append(nodes)
            graph_metadata.append(shard["metadata"][index])

    caption_ids, captions, positives, text_embeddings = [], [], [], []
    for path in sorted((PROCESSED / "texts" / split).glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        caption_ids.extend(shard["caption_ids"])
        captions.extend(shard["captions"])
        positives.extend(shard["positive_image_ids"])
        text_embeddings.append(shard["text_embeddings"])
    assert len(image_ids) == len(set(image_ids))
    assert len(caption_ids) == len(set(caption_ids))
    image_position = {image_id: index for index, image_id in enumerate(image_ids)}
    targets = torch.tensor([image_position[image_id] for image_id in positives], dtype=torch.int64)
    return {
        "image_ids": image_ids,
        "global_image_embeddings": torch.stack(image_embeddings),
        "graph_nodes": graph_nodes,
        "graph_metadata": graph_metadata,
        "caption_ids": caption_ids,
        "captions": captions,
        "positive_image_ids": positives,
        "text_embeddings": torch.cat(text_embeddings),
        "targets": targets,
    }


def hard_negatives(images: torch.Tensor, texts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    images = F.normalize(images.float(), dim=-1).to(DEVICE)
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), 512):
            query = F.normalize(texts[start : start + 512].float(), dim=-1).to(DEVICE)
            scores = query @ images.T
            local_targets = targets[start : start + len(query)].to(DEVICE)
            scores[torch.arange(len(query), device=DEVICE), local_targets] = -torch.inf
            result.append(scores.topk(TOP_K, dim=1).indices.cpu())
            print(f"hard negatives {start + len(query):,}/{len(texts):,}", flush=True)
    return torch.cat(result)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = {"schema_version": 1, "hard_negative_k": TOP_K, "splits": {}}
    for split, expected_images, expected_texts in (
        ("train", 29783, 148915),
        ("val", 1000, 5000),
        ("test", 1000, 5000),
    ):
        data = load_split(split)
        assert len(data["image_ids"]) == expected_images
        assert len(data["caption_ids"]) == expected_texts
        assert data["global_image_embeddings"].shape == (expected_images, 512)
        assert data["text_embeddings"].shape == (expected_texts, 512)
        audit["splits"][split] = {
            "images": expected_images,
            "captions": expected_texts,
            "graph_nodes": sum(len(nodes) for nodes in data["graph_nodes"]),
        }
        if split == "train":
            negative_path = OUTPUT / "train_hard_negatives.pt"
            if not negative_path.exists():
                negatives = hard_negatives(
                    data["global_image_embeddings"], data["text_embeddings"], data["targets"]
                )
                assert negatives.shape == (expected_texts, TOP_K)
                assert not torch.eq(negatives, data["targets"][:, None]).any()
                atomic_save(
                    {
                        "schema_version": 1,
                        "top_k": TOP_K,
                        "caption_ids": data["caption_ids"],
                        "image_ids": data["image_ids"],
                        "targets": data["targets"],
                        "negative_indices": negatives,
                    },
                    negative_path,
                )
        print(split, audit["splits"][split], flush=True)
    temporary = OUTPUT / "audit.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "audit.json")
    print("PREPARE_COMPLETE", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
