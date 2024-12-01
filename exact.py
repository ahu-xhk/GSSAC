import networkx as nx
import time

def dfs(node, graph, visited, current_subgraph, subgraphs, k):
    visited.add(node)
    current_subgraph.add(node)
    subgraphs.add(frozenset(current_subgraph))
    for neighbor in graph.neighbors(node):
        if neighbor not in current_subgraph and graph.degree(neighbor) >= k:
            dfs(neighbor, graph, visited, current_subgraph, subgraphs, k)
    current_subgraph.remove(node)
    visited.remove(node)


def find_subgraphs(graph, uq, k):
    visited = set()
    subgraphs = set()
    current_subgraph = set()

    dfs(uq, graph, visited, current_subgraph, subgraphs, k)
    return subgraphs

def construct_networkx_graph():
    graph = nx.Graph()
    nodes_list = []

    with open("", "r") as file:
        for edge in file:
            cureline = edge.rstrip("\n").rstrip(" ").split(" ")
            cureline = [int(x) for x in cureline]
            for i in range(1, len(cureline)):
                if cureline[0] < cureline[i]:
                    nodes = [cureline[0], cureline[i]]
                    nodes_tuple = tuple(nodes)
                    nodes_list.append(nodes_tuple)

    graph.add_edges_from(nodes_list)
    return graph