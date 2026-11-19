def build_component_batch(component_rows):
    component_rows = list(dict.fromkeys(int(x) for x in component_rows))
    node_chunks, edge_chunks, relation_chunks, reverse_chunks = [], [], [], []
    batch_chunks, root_globals = [], []

    for batch_id, component in enumerate(component_rows):
        ns, ne = compact['node_ptr'][component:component + 2].tolist()
        es, ee = compact['edge_ptr'][component:component + 2].tolist()
        nodes = compact['node_perm'][ns:ne]
        edges = compact['edge_index'][:, es:ee]
        node_types = compact['node_type'][nodes]
        roots = nodes[node_types == 1]
        assert roots.numel() == 1
        node_chunks.append(nodes)
        edge_chunks.append(edges)
        relation_chunks.append(compact['edge_relation_type'][es:ee])
        reverse_chunks.append(compact['edge_is_reverse'][es:ee])
        batch_chunks.append(torch.full((nodes.numel(),), batch_id, dtype=torch.long))
        root_globals.append(roots.item())

    node_global = torch.cat(node_chunks)
    edge_global = torch.cat(edge_chunks, dim=1)
    global_to_local[node_global] = torch.arange(node_global.numel())
    edge_local = global_to_local[edge_global]
    root_local = global_to_local[torch.tensor(root_globals)]
    assert edge_local.min().item() >= 0

    node_types = compact['node_type'][node_global].long()
    feature_rows = compact['node_feature_row'][node_global]
    x = torch.empty((node_global.numel(), 512), dtype=torch.float32)
    image_mask = node_types == 1
    x[image_mask] = image_features[feature_rows[image_mask]]
    x[~image_mask] = entity_features[feature_rows[~image_mask]]
    edge_attr = relation_type_features[torch.cat(relation_chunks)]
    edge_reverse = torch.cat(reverse_chunks).float().unsqueeze(1)
    node_batch = torch.cat(batch_chunks)
    global_to_local[node_global] = -1

    return {
        'x': x.to(DEVICE),
        'node_type': node_types.to(DEVICE),
        'edge_index': edge_local.to(DEVICE),
        'edge_attr': edge_attr.to(DEVICE),
        'edge_reverse': edge_reverse.to(DEVICE),
        'batch': node_batch.to(DEVICE),
        'root_local': root_local.to(DEVICE),
        'components': component_rows,
    }

sample_batch = build_component_batch([0, 1, 2])
print('Sample nodes:', sample_batch['x'].shape)
print('Sample edges:', sample_batch['edge_index'].shape)
print('Sample image roots:', sample_batch['root_local'].shape)
