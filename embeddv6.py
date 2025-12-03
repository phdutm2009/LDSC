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


# Get Average Degree in neighbors of nodes
def average_degree_feature(graph):
    degrees = dict(nx.degree(graph))
    avg_degree = sum(degrees.values()) / len(degrees)
    avg_degree_list = [avg_degree] * len(graph.nodes)
    return torch.tensor(avg_degree_list).view(-1, 1).float()

# PageRank
def pagerank_feature(graph, alpha=0.85, max_iter=100):
    pagerank_scores = nx.pagerank(graph, alpha=alpha, max_iter=max_iter)
    pagerank_list = [pagerank_scores[node] for node in graph.nodes]
    return torch.tensor(pagerank_list).view(-1, 1).float()

# Jaccard Similarity
def average_similarity(graph):
    avg_similarities = []
    avg_similarities_sorenson = []
    for node in graph.nodes:
        neighbors = set(graph.neighbors(node))
        similarities = []
        similarities_sorenson = []
        for neighbor in neighbors:
            neighbor_neighbors = set(graph.neighbors(neighbor))
            numerator = len(neighbors.intersection(neighbor_neighbors))
            denominator = len(neighbors.union(neighbor_neighbors))
            similarity =  numerator/ denominator
            similarities.append(similarity)
            similarities_sorenson.append(2*numerator / denominator)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        avg_similarities.append(avg_similarity)
        avg_similarities_sorenson.append(sum(similarities_sorenson) / len(similarities_sorenson) if similarities_sorenson else 0)
    return torch.tensor(avg_similarities).view(-1, 1).float(), torch.tensor(avg_similarities_sorenson).view(-1, 1).float()

# Sorenson
def average_similarity_sorenson(graph):
    avg_similarities = []
    for node in graph.nodes:
        neighbors = set(graph.neighbors(node))
        similarities = []
        for neighbor in neighbors:
            neighbor_neighbors = set(graph.neighbors(neighbor))
            similarity = (2 * len(neighbors.intersection(neighbor_neighbors))) / len(neighbors.union(neighbor_neighbors))
            similarities.append(similarity)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        avg_similarities.append(avg_similarity)
    return torch.tensor(avg_similarities).view(-1, 1).float()

# Clusterin Coefficent
def cc(graph):
    clustering_coefficients = nx.clustering(graph)
    c = []
    for node, cc in clustering_coefficients.items():
        c.append(cc)
    return torch.tensor(c).view(-1, 1).float()

def triangle(graph):
    triangles = nx.triangles(graph)
    c = []
    for node, cc in triangles.items():
        c.append(cc)
    return torch.tensor(c).view(-1, 1).float()


def ego(graph):
    egos = []
    for node in graph.nodes():
        egos.append(nx.density(nx.ego_graph(graph, node)))

    return torch.tensor(egos).view(-1, 1).float()
    
def ldv(graph):
    l = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = [graph.degree(neighbor) for neighbor in neighbors]

        # Compute local degree variability (LDV)
        ldv_ = np.std(neighbor_degrees)
        l.append(ldv_)
    return torch.tensor(l).view(-1, 1).float()


def AA(graph):
    output = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        aa_indices = [val for u, v, val in nx.adamic_adar_index(graph, [(node, neighbor) for neighbor in neighbors])]
        output.append(round(np.mean(aa_indices) if aa_indices else 0, 3))
    return torch.tensor(output).view(-1, 1).float()


def average_similarity_node(graph, node):
    neighbors = set(graph.neighbors(node))
    similarities = []
    for neighbor in neighbors:
        neighbor_neighbors = set(graph.neighbors(neighbor))
        similarity = (2 * len(neighbors.intersection(neighbor_neighbors))) / len(neighbors.union(neighbor_neighbors))
        similarities.append(similarity)
    similarities = sum(similarities) / len(neighbors)
    return similarities

# --------------------------------------- End Get Features ----------------------------------------------------------


# --------------------------------------------------------------------------------------------------------
# ------------------------------------ Get Supervised Embeddings -----------------------------------------

def get_embeddings(G: nx.Graph, node: int, neighbors_node: List[int], real_labels: Dict[int, int]) -> Dict[int, Dict]:
    ### 
    # Notice one: Nodes in Graph(G) are in string format
    # 
    # ###

    node = str(node)

    subgraph_nodes = [node] + [str(x) for x in neighbors_node]

    subgraph = G.subgraph(subgraph_nodes)

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


    similarity_info = average_similarity(subgraph)
    average_similarity_arr = similarity_info[0]
    average_similarity_sorenson_arr =similarity_info[1]
    pagerank_feature_arr = pagerank_feature(subgraph)
    triangle_arr = triangle(subgraph)

    ldv_arr = ldv(subgraph)

    data = from_networkx(subgraph)

    data.x = torch.cat([
        degree(data.edge_index[0]).view(-1, 1).float(),
        # average_degree_feature_arr,
        average_similarity_arr,
        average_similarity_sorenson_arr,
        # cc_arr,
        triangle_arr,
        ldv_arr,
        pagerank_feature_arr,
    ], dim=1)


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


    data.y = torch.tensor(subgraph_labels_ordered) #Load labels in pytorch

    num_classes = len(set(subgraph_labels_ordered)) # number of labels

    epoch = 10
    # Get Train model
    model = GraphSAGE(in_channels=data.num_features, hidden_channels=16, out_channels=num_classes)
    model.fit(data, epoch)
    model.eval()
    torch.save(model.state_dict(), 'model_state.pth')

    with torch.no_grad():
        latent_representations = model(data.x, data.edge_index)

    nodes = [int(node) for node in subgraph.nodes()]
    embeddings_dict = dict(zip(nodes, latent_representations))

    return embeddings_dict
