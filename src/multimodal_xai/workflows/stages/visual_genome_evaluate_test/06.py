from multimodal_xai.visual_genome.model import EdgeAwareGAT

model = EdgeAwareGAT().to(DEVICE)
with torch.no_grad():
    sample_graph_embeddings = model.encode_graph(sample_batch)
assert sample_graph_embeddings.shape == (3, 128)
print('GAT forward pass: OK', sample_graph_embeddings.shape)
