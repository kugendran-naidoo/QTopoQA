import torch, numpy as np

data = torch.load("graph_data/6W40.pt")

print("node feature shape:", data.x.shape)          # expect (num_nodes, 172)
print("first node feature:", data.x[0].numpy())

# Split the 172-D vector to check components:
basic = data.x[:, :32]                              # classical residue features
topo  = data.x[:, 32:32+140]                        # persistent-homology stats

print("basic feature dims:", basic.shape)
print("topological feature dims:", topo.shape)
print("sample topo stats:", topo[0, :20])

