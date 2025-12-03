from colorama import Fore, Style
import networkx as nx
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import community as c
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import winsound
from collections import  deque, Counter
from numpy import loadtxt
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from dgl.nn import GraphConv
from typing import Dict, Tuple, List
from collections import defaultdict
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import f1_score
from operator import itemgetter
from itertools import groupby
import community as c
from scipy.optimize import linear_sum_assignment

from embeddv6 import get_embeddings

#-----------------dataset choosing------------------------
#---------------------------------------------------------
LFR_I = 4
mu = 7
name = f'LFR{LFR_I}/LFR{LFR_I}_0{mu}'
f = open(f'LFR/{name}.txt', 'r')
# Read the data from the text file
with f as file:
    data = [list(map(str, line.split())) for line in file]

# Create a NetworkX graph
G = nx.Graph()

# Add nodes and edges based on the data
for i, neighbors in enumerate(data):
    for neighbor in neighbors:
        G.add_edge(i, neighbor)

merge_flag = 1
NMI_flag = 1


#---------------------------------------------------------
# ---------------- Real Labels ---------------------------
def get_real_label():
    datasets_real = f'LFR/{name}_real_labels.txt'
    with open(datasets_real, 'r') as f:
        line = f.readlines()
        output = []
        for x in line:
            output.append([int(p) for p in x.split()][0])
        return output

if NMI_flag:
    true_labels = get_real_label()




# print(dict(zip(range(1, len(sorted_nodes) + 1), true_labels)))

#---------------------------------------------------------
# -------------------- Functions we Need -----------------

def neighbors(node, G: nx.Graph=G) -> List:
    node_neighbors = list(G.neighbors(str(node)))
    return [int(x) for x in node_neighbors]

def degree(node:int , G:nx.Graph=G) -> int:
    return G.degree(str(node))

def similarity(node1, node2, G:nx.Graph=G):
    neighbors_1 = set(G.neighbors(str(node1)))
    neighbors_2 = set(G.neighbors(str(node2)))
    return len(neighbors_1.intersection(neighbors_2))/len(neighbors_1.union(neighbors_2))

#-------------------------------------------------------------
#---------------------- 1.Get Score Phase ------------------------

nodes = [int(x) for x in G.nodes()]

degrees_nodes = np.array([degree(x) for x in nodes])

threshold = np.ceil(np.std(degrees_nodes))

print('Standard Deviation is: ', threshold)
print('Median is: ', np.median(degrees_nodes))
print('Mean is: ', np.mean(degrees_nodes))


start_time = time()

embed_score = {node: {'label': set(), 'score': 0} for node in nodes}


def cc(node:int, graph:nx.Graph=G) -> float: # Local Clustering Coefficient

    neighbors_node = neighbors(node)
    k = len(neighbors_node)

    if k < 2:
        return 0.0

    possible_edges = k * (k - 1) / 2
    actual_edges = len([(i, j) for i in neighbors_node for j in neighbors_node if i < j and graph.has_edge(str(i), str(j))])

    clustering_coefficient = actual_edges / possible_edges
    return clustering_coefficient

sim = defaultdict(int)
def find_score(node: int) -> float:
    neighbors_node = neighbors(node)
    degree_node = degree(node)
    if degree_node < threshold:
        return degree_node
    degrees_sum = []
    similarity_sum = []
    degree_with_neighbors_one = 0
    for neig in neighbors_node:
        degree_neig = degree(neig)
        if degree_neig == 1:
            degrees_sum.append(degree_neig)
            degree_with_neighbors_one += 1   
        else:
            degrees_sum.append(degree(neig))
            similarity_sum.append(similarity(node, neig))
    if degree_with_neighbors_one > 0:
        similarity_sum.append(degree_with_neighbors_one * (2.71 ** cc(node)))

    degrees_sum = sum(degrees_sum) / degree_node
    similarity_sum = sum(similarity_sum) / degree_node
    sim[node] = similarity_sum
    return degree_node + degrees_sum + (2 * similarity_sum)

nodes_scores = {node: find_score(node) for node in nodes}
scores = sorted(nodes_scores.items(), key=lambda x: x[1], reverse=True)
sorted_node_score = {node: score for node, score in scores}
del nodes_scores
del scores
del nodes
sorted_nodes = sorted_node_score.keys()
print(Fore.RED + '[INFO-TIME] Get Score of Nodes time is: ', time() - start_time, Style.RESET_ALL)

# ----------------------------------------------------------------------------
# ---------------------------- Functions --------------------------------------
def add_label(node: int, label: int, embed_score=embed_score) -> None:
    embed_score[node]['label'].add(label)

def set_label(node: int, label: set, embed_score: Dict=embed_score) -> None:
    if type(label) == set:
        embed_score[node]['label'] = label
    elif type(label) == int:
        embed_score[node]['label'] = {label}
    else:
        raise "You must provide a label of type set or int for the node."

def get_labels(node: int, embed_score: Dict=embed_score) -> set:
    return embed_score[node]['label']

def get_label(node: int, embed_score: Dict=embed_score) -> int:
    return list(embed_score[node]['label'])[0]

# find score of nodes
def get_score(node: int, sorted_node_score:Dict=sorted_node_score):
    return sorted_node_score[node]

def calculate_avergae_similarity(node:int) -> float:
    neighbors_node = neighbors(node)
    similarites = 0
    for neighbor in neighbors_node:
        similarites += similarity(node, neighbor)
    return similarites / len(neighbors_node)

def get_average_similarty(node:int, sim:Dict=sim) -> float:
    if node in sim.keys():
        return sim[node]
    else:
        avg = calculate_avergae_similarity(node)
        sim[node] = avg
        return avg

# Output => node: {label1, label2, ....}
def get_communites() -> Dict[int, set]:
    return {node:get_labels(node) for node in sorted_nodes}

def predicted_labels():
    communities = get_communites()
    return [list(label)[0] for node, label in sorted(communities.items(), key=lambda x: x[0])]


#-------------------------------------------------------------
#---------------------- 2.Diffuse Phase ------------------------

core_nodes = dict()
start_time = time()
def diffuse(core_node, label):

    add_label(core_node, label)
    neighbors_core_node = neighbors(core_node)
    receive_neighbors_label = [core_node] # Which neighboring nodes that may receive a label.
    core_node_avg_sim = get_average_similarty(core_node)
    # Diffuse Label to Neighbors of core node
    for neighbor in neighbors_core_node:
        if neighbor not in core_nodes.keys():
            if (similarity(core_node, neighbor) > core_node_avg_sim or degree(neighbor) == 1) and get_score(node) > get_score(neighbor):
            # if (similarity(core_node, neighbor) > 0 or degree(neighbor) == 1) and get_score(node) > get_score(neighbor):
                receive_neighbors_label.append(neighbor)
                add_label(neighbor, label)
    
    # Diffuse Label to Second neighbors of core node
    for neighbor in receive_neighbors_label:
        neighbor_avg_sim = get_average_similarty(neighbor)
        neighbors_neighbor = neighbors(neighbor)
        for second_neighbor_of_core_node in neighbors_neighbor:
            if second_neighbor_of_core_node not in core_nodes.keys():
                if second_neighbor_of_core_node not in receive_neighbors_label:
                    if (similarity(neighbor, second_neighbor_of_core_node) >= neighbor_avg_sim or degree(second_neighbor_of_core_node) == 1) and  get_score(neighbor) > get_score(second_neighbor_of_core_node):
                    # if (similarity(neighbor, second_neighbor_of_core_node) >= 0 or degree(second_neighbor_of_core_node) == 1) and  get_score(neighbor) > get_score(second_neighbor_of_core_node):
                        add_label(second_neighbor_of_core_node, label)
    

def check_neighbors_label(node, G:nx.Graph=G):
    neighbors_node = neighbors(node)
    neighbors_label = [list(get_labels(node)) for node in neighbors_node if len(list(get_labels(node))) > 0]
    if len(neighbors_label) >= (degree(node) / 2):
        scores_labels = dict()
        for neighbor in neighbors_node:
            labels = list(get_labels(neighbor))
            if len(labels) == 1:
                if labels[0] in scores_labels.keys():
                    scores_labels[labels[0]] += 1
                else:
                    scores_labels[labels[0]] = 1
            elif len(labels) == 0:
                if 'None' in scores_labels.keys():
                    scores_labels['None'] += 1
                else:
                    scores_labels['None'] = 1
            else:
                score = 1 / len(labels)
                for label in labels:
                    if label in scores_labels.keys():
                        scores_labels[label] += score
                    else:
                        scores_labels[label] = score
        
        label = max(scores_labels.items(), key=lambda x:x[1])[0]
        if label == 'None':
            return "YES"
        else:
            return label        
    return "YES"

label_counter = 0
nodes_small_degree = []
graph_parts = list(nx.connected_components(G))
for node in sorted_nodes:
    if degree(node) == 0:
        add_label(node, -1)
    elif len(get_labels(node)) == 0:
        label = check_neighbors_label(node)
        if len(graph_parts)==1:
            if label == "YES":
                if degree(node) > 2:
                    diffuse(node, label_counter)
                    core_nodes[node] = label_counter
                    label_counter += 1
                else:
                    nodes_small_degree.append(node)
            else:
                add_label(node, label)
        else:
            if label == "YES":
                diffuse(node, label_counter)
                core_nodes[node] = label_counter
                label_counter += 1
            else:
                add_label(node, label)

print(Fore.RED + '[INFO-TIME] Diffuse Pahse time is: ', time() - start_time, Style.RESET_ALL)


#-------------------------------------------------------------
#---------------------- 3.OverLapp Phase ------------------------
#3.1 => detecting Overlapp Nodes:

label_core_node = dict()
for node, label in core_nodes.items():
    label_core_node[label] = node

start_time = time()
overlapp_nodes = [node for node in embed_score.keys() if len(get_labels(node)) > 1]
print(Fore.GREEN + "Number of overlapp nodes Before run OverLapp function: ", len(overlapp_nodes), Style.RESET_ALL)  
def find_overlapp_label():
    for node in overlapp_nodes:
        neighbors_node = [x for x in neighbors(node) if len(list(get_labels(x))) > 0]
        community_situation = defaultdict(int)
        for neighbor in neighbors_node:
            label_neighbor = list(get_labels(neighbor))
            if len(label_neighbor) == 1:
                # community_situation.setdefault(label_neighbor[0], 1) += 1
                community_situation[label_neighbor[0]] += 1
            
            else:
                score = 1 / len(label_neighbor)
                for l in label_neighbor:
                    # community_situation.setdefault(l, 1) += score
                    community_situation[l] += score
        candidate_label = max(community_situation.items(), key=lambda x:x[1])[0]
        set_label(node, candidate_label)



find_overlapp_label()
print(Fore.RED + '[INFO-TIME] Overlapp Pahse time is: ', time() - start_time, Style.RESET_ALL)
overlapp_nodes = [node for node in embed_score.keys() if len(get_labels(node)) > 1]
print(Fore.GREEN + "Number of overlapp nodes After run OverLapp function: ", len(overlapp_nodes), Style.RESET_ALL)  



start_time = time()
new_small_degree = []
# ------------------------ Labels Degree 2 and 1 ----------------------------------

while len(nodes_small_degree) > 0:
    for node in nodes_small_degree:
        label_neighbors = dict()
        for neighbor in neighbors(node):
            if type(get_labels(neighbor)) is int:
                label_neighbors[neighbor] = get_labels(neighbor)
            elif len(list(get_labels(neighbor))) == 1:
                label_neighbors[neighbor] = get_label(neighbor)
            
        if len(label_neighbors.keys()) == 1:
            set_label(node, {list(label_neighbors.values())[0]})
            # set_label(node, list(label_neighbors.values())[0])
        elif len(label_neighbors.keys()) > 1:
            candidate_label = get_label(max({n: get_score(n) for n in neighbors(node)}.items(), key=lambda x: x[1])[0])
            # candidate_label = get_label(max({n: degree(n) for n in neighbors(node)}.items(), key=lambda x: x[1])[0])
            set_label(node, {candidate_label})
        else:
            new_small_degree.append(node)
    nodes_small_degree = new_small_degree.copy()
    new_small_degree = []
        
# print(len(nodes_small_degree))

print(Fore.RED + '[INFO-TIME] Degree 2 and 1 Detection Phase time is: ', time() - start_time, Style.RESET_ALL)


def CALCULAT_PERFORMANCE():
    nodes_temp = G.nodes()
    if NMI_flag == 1:
        real_labels = loadtxt(f"./LFR/{name}_real_labels.txt", comments="#", delimiter="\t", unpack=False)

        detected_labels = predicted_labels()

        detected_labels = np.array(detected_labels)
        real_labels = np.array([int(x) for x in real_labels])
        nmi_score = normalized_mutual_info_score(real_labels, detected_labels)

        print(Fore.GREEN + 'Number of Communities is: ', len(set(detected_labels)))
        print('Number of Real Communities is: ', len(set(real_labels)), Style.RESET_ALL)

        print(f'NMI {name} is: {nmi_score:.4f}')
    return [int(x) for x in nodes_temp]



nodes_temp = CALCULAT_PERFORMANCE()
nodes_sorted_initial = sorted(nodes_temp)
if NMI_flag:
    true_labels_dict = dict(zip(nodes_sorted_initial, true_labels))


#--------------------------------------------------------------------------------
# --------------------------- Detecting Peripheral Nodes ------------------------
peripheral_nodes = []
nodes_without_core_nodes = list(set(sorted_nodes) - set(core_nodes.keys()))
for node in nodes_without_core_nodes:
    neighbors_node = neighbors(node)
    label_neighbors = [get_label(n) for n in neighbors_node]
    if len(set(label_neighbors)) > 1:
        counter_labels = Counter(label_neighbors)
        most_common = counter_labels.most_common(2)
        maximum = most_common[0]
        second_maximum = most_common[1]
        if maximum[0] == get_label(node):
            if second_maximum[1] > maximum[1]/2:
                peripheral_nodes.append(node)
        else:
            peripheral_nodes.append(node)


print(Fore.GREEN + 'Number of Core Nodes is: ', len(core_nodes.keys()))
print('Number of Peripheral Nodes is: ', len(peripheral_nodes), Style.RESET_ALL)


def find_distances(embed_node1: torch.tensor, embed_node2: torch.tensor):
    distance = torch.norm(embed_node1 - embed_node2, p=2) 
    return distance, type(distance)


def frequency_local(node):
    neighbors_node = neighbors(node)
    label_frequency = Counter([list(get_labels(n))[0] for n in neighbors_node])
    max_frequency = max(label_frequency.items(), key=lambda x:x[1])[0]
    if label_frequency[max_frequency] > label_frequency[get_label(node)]:
        set_label(node, {max_frequency})

if NMI_flag:
    def frequency():
        for node in peripheral_nodes:
            if str(node) in nodes_temp:
                neighbors_node = [str(x) for x in neighbors(node) if str(x) in nodes_temp]
                distances = get_embeddings(G, node, neighbors_node, true_labels_dict)
                if distances == 'NO': # There is only one label in its neighborhood of real labels
                    frequency_local(node)
                else:
                    embedd_node = distances[node]
                    distance_neighbors = {neighbor:find_distances(embedd_node, distances[neighbor]) for neighbor in neighbors(node) if neighbor not in peripheral_nodes}
                    if len(distance_neighbors.keys()) > 0:
                        min_distance = min(distance_neighbors.items(), key=lambda x:x[1])[0]
                        set_label(node, get_label(min_distance))
                    else:
                        frequency_local(node)
            else:
                frequency_local(node)
else:
    def frequency():
        for node in peripheral_nodes:
            frequency_local(node)



start_time = time()
frequency()
print(Fore.RED + '[INFO-TIME] Frequency time is: ', time() - start_time, Style.RESET_ALL)
# exit()
# ------------------------------------------------------------------------------
# ------------------------------------ Update Label ----------------------------

def update_labels():
    for node in sorted_nodes:
        neighbors_node = neighbors(node)
        neighbor_communities = [get_label(neighbor) for neighbor in neighbors_node]
        if neighbor_communities:
            most_common_community = max(set(neighbor_communities), key=neighbor_communities.count)
            set_label(node, {most_common_community})

start_time = time()
# update_labels()
print(Fore.RED + '[INFO-TIME] Update Label time is: ', time() - start_time, Style.RESET_ALL)

# ----------------------------------- SHOW RESULTS ----------------------------
# -----------------------------------------------------------------------------
CALCULAT_PERFORMANCE()


# -------------------------------------------------------------------------------
# ------------------------- Community Density -----------------------------------
# From paper 1 in core nodes: Locating Structural Centers: A Density-Based Clustering Method for Community Detection
def community_density(community: list) -> float:
    edges = []
    for node in community:
        neighbors_node = neighbors(node)
        for neighbor in neighbors_node:
            if neighbor in community:
                if (node, neighbor) not in edges or (neighbor, node) not in edges:
                    edges.append((node, neighbor))
    degrees_community = sum([degree(node) for node in community])
    return len(edges) / degrees_community


def community_dict():
    communities_tmp = dict()
    for node in sorted_nodes:
        label = get_label(node)
        if label not in communities_tmp.keys():
            communities_tmp[label] = [node]
        else:
            communities_tmp[label].append(node)
    return communities_tmp


# communities_dict = community_dict()
# # # print(communities_dict)

# for community, l in communities_dict.items():
#     print(f'Density for community {community} is {community_density(l)}')



################# Merge : ***********************************************
# for node in sorted_nodes:
#     print(node, embed_score[node]['label'])

def communities_labels_set(labels):
    output = {i: [] for i in set(labels)}
    for node in sorted_nodes:
        output[get_label(node)].append(node)
    return output

detected_labels = [get_label(node) for node in sorted_nodes]
start_time = time()
new_communites = communities_labels_set(detected_labels)
print(Fore.RED + '[TIME-INFO] time get new_communities : ', time() - start_time, Style.RESET_ALL)

def find_small_communities(new_communites):
    len_communities = [len(val) for _,val in new_communites.items()]
    max_community = max(len_communities)
    len_communities = [x for x in len_communities if x != max_community]
    average = sum(len_communities) / len(len_communities)
    small_communities = [val for _,val in new_communites.items() if len(val) <= average]
    return small_communities


def find_most_important_neighbor(community):
    output = []
    for node in community:
        output.append((node, get_score(node)))

    return max(output,key=lambda x:x[1])[0]

def merge(new_communites):
    small_communities = find_small_communities(new_communites)
    for community in small_communities:
        candidate_node = find_most_important_neighbor(community)
        x = max([(node , degree(node)) for node in neighbors(candidate_node)],key=lambda x:x[1])[0]
        most_important_neighbors = get_label(x)
        if get_label(candidate_node) != most_important_neighbors:
            if get_score(x) > get_score(candidate_node) or similarity(candidate_node, x) > 0:
                for node in community:
                    set_label(node, {most_important_neighbors})


start_time = time()
if len(set(detected_labels)) > 2:
    merge(new_communites)

print(Fore.RED + '[TIME-INFO] time merge community : ', time() - start_time, Style.RESET_ALL)





CALCULAT_PERFORMANCE()

#--------------------------------------------------------------------------------
# --------------------------- Detecting Peripheral Nodes ------------------------
# peripheral_nodes = []
# nodes_without_core_nodes = list(set(sorted_nodes) - set(core_nodes.keys()))
# for node in nodes_without_core_nodes:
#     neighbors_node = neighbors(node)
#     counter_labels = Counter([get_label(n) for n in neighbors_node])
#     if counter_labels[get_label(node)] < (degree(node) * (70/100)):
#         peripheral_nodes.append(node)

peripheral_nodes = []
nodes_without_core_nodes = list(set(sorted_nodes) - set(core_nodes.keys()))
for node in nodes_without_core_nodes:
    neighbors_node = neighbors(node)
    label_neighbors = [get_label(n) for n in neighbors_node]
    if len(set(label_neighbors)) > 1:
        counter_labels = Counter(label_neighbors)
        most_common = counter_labels.most_common(2)
        maximum = most_common[0]
        second_maximum = most_common[1]
        if maximum[0] == get_label(node):
            if second_maximum[1] > maximum[1]/2:
                peripheral_nodes.append(node)
        else:
            peripheral_nodes.append(node)


print(Fore.GREEN + 'Number of Core Nodes is: ', len(core_nodes.keys()))
print('Number of Peripheral Nodes is: ', len(peripheral_nodes), Style.RESET_ALL)

start_time = time()
frequency()
# frequency_degree()
print(Fore.RED + '[INFO-TIME] Frequency time is: ', time() - start_time, Style.RESET_ALL)


CALCULAT_PERFORMANCE()

start_time = time()
update_labels()
print(Fore.RED + '[INFO-TIME] Update Label time is: ', time() - start_time, Style.RESET_ALL)

CALCULAT_PERFORMANCE()
