"""Render the submitted sample scene graph for Visual Genome."""
from collections import Counter, defaultdict
from pathlib import Path
import math


def render_sample(root: Path, output: Path):
    import duckdb
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import networkx as nx


    graph = root.expanduser().resolve() / "dataset/visual_genome/graph/split_v2"
    image_id = "vg:image:61512"
    connection = duckdb.connect(database=":memory:")
    nodes = connection.execute(
        """
        SELECT node_id, node_type, label_norm
        FROM read_parquet(?)
        WHERE context_id=?
        """,
        [str(graph / "nodes.parquet"), image_id],
    ).fetchall()
    edges = connection.execute(
        """
        SELECT source_node, relation, target_node, edge_family
        FROM read_parquet(?)
        WHERE context_id=?
        """,
        [str(graph / "edges.parquet"), image_id],
    ).fetchall()
    connection.close()

    node_type = {node_id: kind for node_id, kind, _ in nodes}
    raw_label = {node_id: label for node_id, _, label in nodes}
    label_totals = Counter(raw_label.values())
    label_seen = defaultdict(int)
    labels = {}
    for node_id, _, label in sorted(nodes):
        label_seen[label] += 1
        labels[node_id] = (
            label if label_totals[label] == 1
            else f"{label} {label_seen[label]}"
        )
    labels[image_id] = "IMAGE 61512"

    entities = sorted(
        [node_id for node_id, kind, _ in nodes if kind == "entity"],
        key=lambda node_id: (labels[node_id], node_id),
    )
    positions = {image_id: (0.0, 0.0)}
    radius = 5.3
    for index, node_id in enumerate(entities):
        angle = math.pi / 2 - 2 * math.pi * index / len(entities)
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    fig, axis = plt.subplots(figsize=(16, 15))

    contains_edges = [(u, v) for u, _, v, family in edges if family == "image_entity"]
    semantic_edges = [(u, relation, v) for u, relation, v, family in edges if family == "entity_relation"]

    nx.draw_networkx_edges(
        nx.DiGraph(contains_edges), positions, edgelist=contains_edges,
        edge_color="#b0bec5", style="dashed", width=1.2,
        arrows=True, arrowsize=13, alpha=0.75, ax=axis,
    )

    parallel_count = defaultdict(int)
    for source, relation, target in semantic_edges:
        pair = (source, target)
        parallel_index = parallel_count[pair]
        parallel_count[pair] += 1
        rad = 0.12 + 0.13 * parallel_index
        arrow = FancyArrowPatch(
            positions[source], positions[target],
            arrowstyle="-|>", mutation_scale=17,
            linewidth=2.2, color="#d1495b",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=25, shrinkB=25, zorder=3,
        )
        axis.add_patch(arrow)
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = max(math.hypot(dx, dy), 1e-6)
        offset = 0.22 + 0.22 * parallel_index
        axis.text(
            midpoint_x - dy / length * offset,
            midpoint_y + dx / length * offset,
            relation,
            fontsize=9, color="#9b2226", fontweight="bold",
            ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
            zorder=5,
        )

    nx.draw_networkx_nodes(
        nx.Graph(), positions, nodelist=[image_id], node_size=3600,
        node_color="#90caf9", edgecolors="#263238", linewidths=2.0, ax=axis,
    )
    nx.draw_networkx_nodes(
        nx.Graph(), positions, nodelist=entities, node_size=2100,
        node_color="#ffe0a3", edgecolors="#455a64", linewidths=1.4, ax=axis,
    )
    nx.draw_networkx_labels(
        nx.Graph(), positions, labels=labels, font_size=9,
        font_weight="bold", ax=axis,
    )

    axis.set_title(
        "Full Visual Genome graph component — vg:image:61512\n"
        "18 entities, 18 image–contains edges, 9 semantic relation edges",
        fontsize=17,
    )
    axis.text(
        0.01, 0.015,
        "Dashed gray: image → contains → entity     Red: semantic entity → relation → entity",
        transform=axis.transAxes, fontsize=11,
    )
    axis.set_xlim(-6.8, 6.8)
    axis.set_ylim(-6.5, 6.7)
    axis.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)
