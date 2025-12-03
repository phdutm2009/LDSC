import networkx as nx
import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from typing import List, Dict, Tuple, Set
from torch_geometric.utils import from_networkx, degree
from torch import tensor
from torch_geometric.nn import SAGEConv
from scipy.sparse import csr_matrix

def neighbors(node, list_node_neighbors):
    return list_node_neighbors[node]

def get_subgraph(node, neighbors_node, list_node_neighbors):
    subgraph_nodes = set(neighbors_node + [node])
    subgraph_edges = []

    for n in subgraph_nodes:
        for neighbor in neighbors(n, list_node_neighbors):
            if neighbor in subgraph_nodes:
                subgraph_edges.append((n, neighbor))

    return subgraph_nodes, subgraph_edges



# def pagerank(list_node_neighbors, N, alpha=0.85, max_iter=100, tol=1.0e-6):
#     # Initialize PageRank scores
#     pagerank_scores = np.ones(N) / N
#     out_degree = np.array([len(neighbors(node, list_node_neighbors)) for node in range(N)])

#     for _ in range(max_iter):
#         new_pagerank_scores = np.ones(N) * (1 - alpha) / N + alpha * np.array(
#             [sum(pagerank_scores[j] / out_degree[j] for j in neighbors(i, list_node_neighbors)) for i in range(N)]
#         )
        
#         if np.linalg.norm(new_pagerank_scores - pagerank_scores, 1) < tol:
#             break
        
#         pagerank_scores = new_pagerank_scores
    
#     return pagerank_scores



def pagerank(subgraph_nodes, list_node_neighbors, alpha=0.85, max_iter=100):
    N = len(subgraph_nodes)
    out_degree = [degree(node) for node in subgraph_nodes]
    pagerank_scores = np.ones(N) / N

    for _ in range(max_iter):
        new_pagerank_scores = np.ones(N) * (1 - alpha) / N + alpha * np.array(
            [sum(pagerank_scores[j] / out_degree[j] for j in neighbors(i, list_node_neighbors)) for i in range(N)]
        )

        pagerank_scores = new_pagerank_scores
    return pagerank_scores

def pagerank_feature(subgraph_nodes, list_node_neighbors):
    # Calculate PageRank scores for your graph
    pagerank_scores = pagerank(subgraph_nodes, list_node_neighbors)

    # Convert the scores into a torch tensor
    pagerank_list = [pagerank_scores[node] for node in range(len(list_node_neighbors))]
    pagerank_tensor = torch.tensor(pagerank_list).view(-1, 1).float()

    return pagerank_tensor

# Jaccard Similarity
def average_similarity(graph_nodes, list_node_neighbors):
    avg_similarities = []
    avg_similarities_sorenson = []
    for node in list_node_neighbors:
        neighbors = set(neighbors(node, list_node_neighbors))
        similarities = []
        similarities_sorenson = []
        for neighbor in neighbors:
            neighbor_neighbors = set(neighbors(neighbor, list_node_neighbors))
            numerator = len(neighbors.intersection(neighbor_neighbors))
            denominator = len(neighbors.union(neighbor_neighbors))
            similarity =  numerator/ denominator
            similarities.append(similarity)
            similarities_sorenson.append(2*numerator / denominator)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        avg_similarities.append(avg_similarity)
        avg_similarities_sorenson.append(sum(similarities_sorenson) / len(similarities_sorenson) if similarities_sorenson else 0)
    return torch.tensor(avg_similarities).view(-1, 1).float(), torch.tensor(avg_similarities_sorenson).view(-1, 1).float()




def triangle(subgraph_nodes, list_node_neighbors):
    num_triangles = []
    for node in subgraph_nodes:
        neighbors_node = list_node_neighbors[node]
        num_triangle = 0
        for neighbor in neighbors_node:
            common_neigh = set(neighbors_node).intersection(set(list_node_neighbors[neighbor]))
            for common_node in common_neigh:
                if common_node != node and common_node != neighbor and neighbor in list_node_neighbors[common_node]:
                    num_triangle += 1
        num_triangles.append(num_triangle)

    return torch.tensor(num_triangles).view(-1, 1).float()

    
def ldv(graph_noeds, list_node_neighbors):
    l = []
    for node in graph_noeds:
        neighbors = neighbors(node, list_node_neighbors)
        neighbor_degrees = [len(neighbors(neighbor, list_node_neighbors)) for neighbor in neighbors]

        # Compute local degree variability (LDV)
        ldv_ = np.std(neighbor_degrees)
        l.append(ldv_)
    return torch.tensor(l).view(-1, 1).float()



# --------------------------------------- End Get Features ----------------------------------------------------------



# --------------------------------------------------------------------------------------------------------
# ------------------------------------ Get Supervised Embeddings -----------------------------------------

def get_embeddings(list_node_neighbors, node: int, neighbors_node: List[int], real_labels: Dict[int, int]) -> Dict[int, Dict]:
    ### 
    # Notice one: Nodes in Graph(G) are in string format
    # 
    # ###

    subgraph_nodes = [node] + [x for x in neighbors_node]

    subgraph_nodes, subgraph_edges = get_subgraph(node, list_node_neighbors=list_node_neighbors, neighbors_node=neighbors_node)

    subgraph_labels = []
    for n in subgraph_nodes:
        subgraph_labels.append(real_labels[int(n)])

    set_subgraph_labels = set(subgraph_labels)
    if len(set_subgraph_labels) == 1:
        return 'NO'
    
    tmp_label = dict()
    i = 0
    for l in set_subgraph_labels:
        tmp_label[l] = i
        i += 1

    subgraph_labels_ordered = []
    for l in subgraph_labels:
        subgraph_labels_ordered.append(tmp_label[l])


    similarity_info = average_similarity(subgraph_nodes, list_node_neighbors)
    average_similarity_arr = similarity_info[0]
    average_similarity_sorenson_arr =similarity_info[1]
    pagerank_feature_arr = pagerank_feature(subgraph_nodes)

   
    # cc_arr = cc(G)

    triangle_arr = triangle(subgraph_nodes, list_node_neighbors)

    ldv_arr = ldv(subgraph_nodes, list_node_neighbors)

    
    # Initialize variables
    rows = []
    cols = []


    # Populate the row and column indices for non-zero elements in the adjacency matrix
    for node, neighbors in enumerate(list_node_neighbors):
        for neighbor in neighbors:
            rows.append(node)
            cols.append(neighbor)

    # Create a sparse adjacency matrix using CSR format
    adjacency_matrix_sparse = csr_matrix(([1] * len(rows), (rows, cols)), shape=(len(subgraph_nodes), len(subgraph_nodes)))

    # You can convert it to a dense matrix if needed
    adjacency_matrix_dense = adjacency_matrix_sparse.toarray()

    # Convert the dense adjacency matrix to edge index format
    edge_index = torch.tensor(adjacency_matrix_dense.nonzero(), dtype=torch.long)

    # Assuming you have node labels stored in 'node_labels'
    node_labels = torch.tensor(node_labels, dtype=torch.long)

    x = torch.cat([
        degree(data.edge_index[0]).view(-1, 1).float(),
        average_similarity_arr,
        average_similarity_sorenson_arr,
        triangle_arr,
        ldv_arr,
        pagerank_feature_arr,
    ], dim=1)


    data = Data(x=x, edge_index=edge_index, y=torch.tensor(subgraph_labels_ordered))

    class GraphSAGE(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
            super(GraphSAGE, self).__init__()
            self.out_channels = out_channels
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
            self.conv4 = SAGEConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            x = self.conv4(x, edge_index)
            return x

        def accuracy(pred_y, y):
            return ((pred_y == y).sum() / len(y)).item()

        def fit(self, data, epochs):
            data.train_mask = torch.ones(data.num_nodes, dtype=bool)
            data.test_mask = torch.ones(data.num_nodes, dtype=bool)

            optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.001)
            if self.out_channels == 2:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()

            self.train()
            for epoch in range(epochs + 1):
                optimizer.zero_grad()
                out = self(data.x, data.edge_index)
                if self.out_channels == 2:
                    target = data.y[data.train_mask].clamp(max=1)
                    target = F.one_hot(target, num_classes=2).float()  # One-hot encode the adjusted target labels
                    loss = F.binary_cross_entropy_with_logits(out[data.train_mask], target)
                else:
                    loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()




    num_classes = len(set(subgraph_labels_ordered)) # number of labels

    epoch = 10
    # Get Train model
    model = GraphSAGE(in_channels=data.num_features, hidden_channels=16, out_channels=num_classes)
    model.fit(data, epoch)
    model.eval()
    torch.save(model.state_dict(), 'model_state.pth')

    with torch.no_grad():
        latent_representations = model(data.x, data.edge_index)

    nodes = [int(node) for node in subgraph_nodes]
    embeddings_dict = dict(zip(nodes, latent_representations))

    return embeddings_dict
