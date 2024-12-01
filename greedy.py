import copy
import json
import math
import sys
import time
import random
from greedy_main import Social_Graph, Road_Graph
from generate_embedding import get_embedding_file, generate_user_prompts, compute_semantic_similarity, \
    get_embedding_for_community_keyword_set, get_embedding_for_query_node_keyword_set, cosine_similarity_for_vec


def get_degree(G):

    degree = {}
    for i in G:
        degree[i] = len(G[i])  
    return degree

def core_number(G):
    degrees = get_degree(G)
    nodes = sorted(degrees, key=degrees.get)

    bin_boundaries = [0]
    curr_degree = 0     
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
 
    node_pos = {v: pos for pos, v in enumerate(nodes)}

    core = degrees 

    for v in nodes:
        for u in G[v]:
            if core[u] > core[v]:
                G[u].remove(v)     
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1        


    return core

def mini_degree_of_graph(subgraph):

    mini_degree = int(1e+8)
    for key, value in subgraph.edge.items():
        if len(value) < mini_degree:
            mini_degree = len(value)
            node = key
    return mini_degree, node

def find_minimum_k_core_7(social_graph, k, uq, road_graph):


    flag = 0
    social_subgraph = Social_Graph()
    social_subgraph_neighbors = {uq: [0, 0, 1]}
    previous_size = -1
    loop = 0

    max_count = 0
    min_count = 0
    max_dist_uq = 0
    min_dist_uq = 0
    mean_count = 0
    mean_dist_uq = 0
    eps = 0.00000001

    while True:
        if len(social_subgraph.edge.keys()) == previous_size:
            return social_subgraph, flag
        previous_size = len(social_subgraph.edge.keys())
        neighbors_list_dict = {}
        denominator_count = min_count - max_count
        if denominator_count == 0:
            denominator_count = -max_count - eps

        denominator_dist_uq = min_dist_uq - max_dist_uq
        if denominator_dist_uq == 0:
            denominator_dist_uq = max_dist_uq - eps

        mean_count =  sum([value[0] for key, value in social_subgraph_neighbors.items()]) / len(social_subgraph_neighbors.keys())
        mean_dist_uq =  sum([value[1] for key, value in social_subgraph_neighbors.items()]) / len(social_subgraph_neighbors.keys())

        for key, value in social_subgraph_neighbors.items():
            value_0_coef = value[2]
            tempMD = ((value_0_coef * value[0] - max_count)/denominator_count) + ((value[1] - max_dist_uq)/denominator_dist_uq)
            if mean_count != 0 and mean_dist_uq != 0:
                dev = value[1]/mean_dist_uq - value[0]/mean_count
            else:
                dev = value[1] - value[0]
            neighbors_list_dict.update({key : (round(tempMD, 8), round(dev, 8))})
        neighbors_list = sorted(neighbors_list_dict.items(), key=lambda x: (x[1][0], x[1][1]))


        if len(neighbors_list) != 0:
            curnode = neighbors_list[0][0]
        del social_subgraph_neighbors[curnode]

        temp_neighbor = []
        for i in social_graph.edge[curnode]:
            if i in social_subgraph.edge.keys():
                temp_neighbor.append(i)
                social_subgraph.edge[i].append(curnode)
        social_subgraph.edge.update({curnode: temp_neighbor})

        core_noumber_social_graph = copy.deepcopy(social_subgraph.edge)
        core_n = core_number(core_noumber_social_graph)

        if core_n[uq] >= k:
            flag = 1
            k_core_social_subgraph = Social_Graph()
            k_core_social_subgraph.edge.update({uq: []})
            while True:
                previous_num = len(k_core_social_subgraph.edge.keys())
                temporary_set = set()
                current_node_set = set(k_core_social_subgraph.edge.keys())
                for node in current_node_set:
                    temporary_set.update(set(social_subgraph.edge[node]))
                temporary_set = temporary_set - current_node_set
                for neighbor_node in temporary_set:
                    if core_n[neighbor_node] >= k:
                        temporary_neighbor = []
                        for i in social_subgraph.edge[neighbor_node]:
                            if i in k_core_social_subgraph.edge.keys():
                                temporary_neighbor.append(i)
                                k_core_social_subgraph.edge[i].append(neighbor_node)
                        k_core_social_subgraph.edge.update({neighbor_node: temporary_neighbor})
                if len(k_core_social_subgraph.edge.keys()) == previous_num:
                    break
                mini_degree, mini_node = mini_degree_of_graph(k_core_social_subgraph)
                if mini_degree >= k:
            return k_core_social_subgraph, flag


        max_count = float('-inf')
        min_count = float('inf')
        max_dist_uq = float('-inf')
        min_dist_uq = float('inf')


        for subgraph_node in social_subgraph.edge.keys():
            for subgraph_node_neighbor in social_graph.edge[subgraph_node]:
                if subgraph_node_neighbor not in social_subgraph.edge.keys() and len(social_graph.edge[subgraph_node_neighbor]) >= k:

                    count = len(set(social_graph.edge[subgraph_node_neighbor]).intersection(set(social_subgraph.edge.keys())))

                    if subgraph_node_neighbor in social_subgraph_neighbors.keys():
                        dist_uq = social_subgraph_neighbors[subgraph_node_neighbor][1]
                        hop = social_subgraph_neighbors[subgraph_node_neighbor][2]
                        max_count = max(max_count, count * hop)
                        min_count = min(min_count, count * hop)
                        max_dist_uq = max(max_dist_uq, dist_uq)
                        min_dist_uq = min(min_dist_uq, dist_uq)

                        social_subgraph_neighbors.update({subgraph_node_neighbor: [count, dist_uq, hop]})
                    else:
                        x3 = road_graph.coordinate[social_graph.location[uq]][0] - road_graph.coordinate[social_graph.location[subgraph_node_neighbor]][0]
                        y3 = road_graph.coordinate[social_graph.location[uq]][1] - road_graph.coordinate[social_graph.location[subgraph_node_neighbor]][1]
                        dist_uq = round(math.sqrt(x3 ** 2 + y3 ** 2) * (-1), 8)
                        hop = (1 + (1 / ((loop + 1) ** 2)))
                        max_count = max(max_count, count * hop)
                        min_count = min(min_count, count * hop)
                        max_dist_uq = max(max_dist_uq, dist_uq)
                        min_dist_uq = min(min_dist_uq, dist_uq)

                        social_subgraph_neighbors.update({subgraph_node_neighbor: [count, dist_uq, hop]})
        loop += 1

def compute_keyword_score(subgraph, T, social_graph, uq):
    len_of_T = len(T)
    size_of_subgraph = len(subgraph.edge.keys()) - 1
    T_num = 0
    for w in T:
        w_num = 0
        for key in subgraph.edge.keys():
            if key != uq and w in social_graph.keyword[key]:
                w_num += 1
        T_num += w_num
    f1 = T_num / (len_of_T * size_of_subgraph)
    keyword_of_subgraph = set()
    for key in subgraph.edge.keys():
        if key != uq:
            keyword_of_subgraph.update(social_graph.keyword[key])
    intersection_set = T.intersection(keyword_of_subgraph)
    f2 = len(intersection_set) / len_of_T
    return round(f1 + f2, 8)

def compute_keyword_score_with_embeddings(subgraph, T, social_graph, uq):

    size_of_subgraph = len(subgraph.edge.keys()) - 1
    with open('config.json') as config_file:
        config = json.load(config_file)

    user_embeddings_file = "../04_vectors/" + config['experiment_dataset'] + "_embeddings.json"

    try:
        with open(user_embeddings_file, 'r') as f:
            embedding_dict = json.load(f) 
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        embedding_dict = {}
    set1 = subgraph.edge.keys()
    set2 = {int(element) for element in set(embedding_dict.keys())}
    unknown_set = set1 - set2
    print(unknown_set)
    unknown_dict = {}
    for element in unknown_set:
        unknown_dict.update({element: social_graph.keyword[element]})
    print(unknown_dict)

    if len(unknown_dict) != 0:
        generate_user_prompts(unknown_dict)
        get_embedding_file()

    results = 0.0
    for node in subgraph.edge.keys():
        if node != uq:
            results += compute_semantic_similarity(node, uq)
    embedding_score = results / size_of_subgraph


    len_of_T = len(T)
    T_num = 0
    for w in T:
        w_num = 0
        for key in subgraph.edge.keys():
            if key != uq and w in social_graph.keyword[key]:
                w_num += 1
        T_num += w_num
    f1 = T_num / (len_of_T * size_of_subgraph)


    keyword_of_subgraph = set()
    for key in subgraph.edge.keys():
        if key != uq:
            keyword_of_subgraph.update(social_graph.keyword[key])
    intersection_set = T.intersection(keyword_of_subgraph)
    f2 = len(intersection_set) / len_of_T

    score = ((f1 + f2) + embedding_score) / 2
    return round(score, 8)

def compute_keyword_score_only_embeddings(subgraph, T, social_graph, uq):
    size_of_subgraph = len(subgraph.edge.keys()) - 1
    with open('config.json') as config_file:
        config = json.load(config_file)
    user_embeddings_file = "../04_vectors/" + config['experiment_dataset'] + "_embeddings.json"
    try:
        with open(user_embeddings_file, 'r') as f:
            embedding_dict = json.load(f) 
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        embedding_dict = {}

    set1 = subgraph.edge.keys()
    set2 = {int(element) for element in set(embedding_dict.keys())}
    unknown_set = set1 - set2
    unknown_dict = {}
    for element in unknown_set:
        unknown_dict.update({element: social_graph.keyword[element]})
    if len(unknown_dict) != 0:
        generate_user_prompts(unknown_dict)

        get_embedding_file()

    results = 0.0
    for node in subgraph.edge.keys():
        if node != uq:
            results += compute_semantic_similarity(node, uq)


    return round(results / size_of_subgraph, 8)


def compute_keyword_score_entire_community(subgraph, T, social_graph, uq):
    community_keywords = set()
    for key in subgraph.edge.keys():
        if key != uq:
            community_keywords.update(social_graph.keyword[key])

    community_keyword_embedding = get_embedding_for_community_keyword_set(community_keywords)

    query_node_embedding = get_embedding_for_query_node_keyword_set(uq, social_graph)

    return cosine_similarity_for_vec(community_keyword_embedding, query_node_embedding)

def compute_keyword_score_max_single_keyword(subgraph, T, social_graph, uq):
    with open('config.json') as config_file:
        config = json.load(config_file)

    user_embeddings_file = "../04_vectors/" + config['experiment_dataset'] + "single_keyword_embeddings.json"

    try:
        with open(user_embeddings_file, 'r') as f:
            embedding_dict = json.load(f)  
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        embedding_dict = {}

    set1 = subgraph.edge.keys()
    set2 = {int(element) for element in set(embedding_dict.keys())}
    unknown_set = set1 - set2
    unknown_dict = {}
    for element in unknown_set:
        unknown_dict.update({element: social_graph.keyword[element]})
    if len(unknown_dict) != 0:
        generate_user_prompts(unknown_dict)
        get_embedding_file()

    results = -1
    for node in subgraph.edge.keys():
        if node != uq:
            results += compute_semantic_similarity(node, uq)

    results = results / size_of_subgraph

    pass

def shortest_travel_time(road_graph, start_node, end_node, index, flag, remainder, timetable):
    if start_node == end_node:
        return 0
    d_length, p_length, f_length, opentable, closetable = {}, {}, {}, {}, {}
    parent = {}
    neighbors_of_start_node = set(road_graph.edge[start_node]) - {start_node}
    if len(neighbors_of_start_node) == 0:
        x0 = road_graph.coordinate[start_node][0] - road_graph.coordinate[end_node][0]
        y0 = road_graph.coordinate[start_node][1] - road_graph.coordinate[end_node][1]
        x0 = x0 * 100000
        y0 = y0 * 100000
        return round(math.sqrt(x0 ** 2 + y0 ** 2), 8)
    for i in neighbors_of_start_node:
        x1 = road_graph.coordinate[start_node][0] - road_graph.coordinate[i][0]
        y1 = road_graph.coordinate[start_node][1] - road_graph.coordinate[i][1]
        x2 = road_graph.coordinate[i][0] - road_graph.coordinate[end_node][0]
        y2 = road_graph.coordinate[i][1] - road_graph.coordinate[end_node][1]
        edge_temp = (min(i, start_node), max(i, start_node))
        if flag == 0:
            time_temp = road_graph.time[edge_temp][index]
        else:
            if edge_temp in timetable:
                time_temp = timetable[edge_temp]
            else:
                slope = (road_graph.time[edge_temp][index + 1] - road_graph.time[edge_temp][index]) / 5
                time_temp = road_graph.time[edge_temp][index] + remainder * slope
                timetable.update({edge_temp: time_temp})
        p_length[i] = time_temp
        edge_temp_2 = (min(i, end_node), max(i, end_node))
        d_length[i] = math.sqrt(x2 ** 2 + y2 ** 2) * 6 + 10
        f_length[i] = p_length[i] + d_length[i]
        opentable[i] = f_length[i]
        parent[i] = start_node
    while True:
        opentable = sorted(opentable.items(), key=lambda x: x[1])
        opentable = dict(opentable)
        min_node = list(opentable.keys())[0]
        closetable[min_node] = opentable[min_node]
        if min_node == end_node:
            break
        opentable.pop(min_node)
        neighbors_of_min_node = set(road_graph.edge[min_node]) - {start_node} - {min_node}
        for i in neighbors_of_min_node:
            x1 = road_graph.coordinate[min_node][0] - road_graph.coordinate[i][0]
            y1 = road_graph.coordinate[min_node][1] - road_graph.coordinate[i][1]
            x2 = road_graph.coordinate[i][0] - road_graph.coordinate[end_node][0]
            y2 = road_graph.coordinate[i][1] - road_graph.coordinate[end_node][1]
            edge_temp = (min(i, min_node), max(i, min_node))
            if flag == 0:
                time_temp = road_graph.time[edge_temp][index]
            else:
                if edge_temp in timetable:
                    time_temp = timetable[edge_temp]
                else:
                    slope = (road_graph.time[edge_temp][index + 1] - road_graph.time[edge_temp][index]) / 5
                    time_temp = road_graph.time[edge_temp][index] + remainder * slope
                    timetable.update({edge_temp: time_temp})
            upgrade_p = p_length[min_node] + time_temp
            edge_temp_2 = (min(i, end_node), max(i, end_node))
            d_length[i] = math.sqrt(x2 ** 2 + y2 ** 2) * 6 + 10
            upgrade = upgrade_p + d_length[i]
            if i in opentable.keys() and upgrade < f_length[i]:
                p_length[i] = upgrade_p
                f_length[i] = upgrade
                parent[i] = min_node
            if i in closetable.keys() and upgrade < f_length[i]:
                p_length[i] = upgrade_p
                closetable.pop(i)
                opentable[i] = upgrade
                f_length[i] = upgrade
                parent[i] = min_node
            if i not in opentable.keys() and i not in closetable.keys():
                p_length[i] = upgrade_p
                f_length[i] = upgrade
                opentable[i] = f_length[i]
                parent[i] = min_node
    pathlength = closetable[end_node]
    return round(pathlength, 8)


def compute_average_time(subgraph, road_graph, social_graph, t, vr, timetable, road_shortest_timetable):

    flag = 0  
    if t % 5 != 0:
        flag = 1
    index = t // 5
    remainder = t % 5
    cumulate_time = 0
    for key in subgraph.edge.keys():
        loc_of_node = social_graph.location[key]
        road_temp = (min(loc_of_node, vr), max(loc_of_node, vr))
        if road_temp in road_shortest_timetable:
            time_of_node = road_shortest_timetable[road_temp]
        else:
            time_of_node = shortest_travel_time(road_graph, loc_of_node, vr, index, flag, remainder, timetable)
            road_shortest_timetable.update({road_temp: time_of_node})
        cumulate_time += time_of_node

    return round(-(cumulate_time / len(subgraph.edge.keys())), 8)


def judge_dominance(subgraph, node, social_graph, T, road_graph, t, vr, candidate_social_subgraph, uq, timetable, road_shortest_timetable):
    temp_social_subgraph = copy.deepcopy(subgraph)
    temp_neighbor = []
    for i in social_graph.edge[node]:
        if i in temp_social_subgraph.edge.keys():
            temp_neighbor.append(i)
            temp_social_subgraph.edge[i].append(node)
    temp_social_subgraph.edge.update({node: temp_neighbor})
    temp_social_subgraph.keyword_score = compute_keyword_score_entire_community(temp_social_subgraph, T, social_graph, uq)
    temp_social_subgraph.time_score = compute_average_time(temp_social_subgraph, road_graph, social_graph, t, vr, timetable, road_shortest_timetable)

    non_dominance = 0
    if temp_social_subgraph.keyword_score > subgraph.keyword_score or temp_social_subgraph.time_score > subgraph.time_score:
        candidate_social_subgraph.update({node: [temp_social_subgraph.keyword_score, temp_social_subgraph.time_score]})
        non_dominance = 1
    else:
        non_dominance = 0
    return non_dominance


def sort_candidate_neighbors_new(social_subgraph, candidate_social_subgraph):

    temp_dict = dict()
    keyword_score_max = float('-inf')
    keyword_score_min = float('inf')
    time_score_max = float('-inf')
    time_score_min = float('inf')
    mean_keyword_score = 0
    mean_time_score = 0
    eps = 0.00000001
    for key, value in candidate_social_subgraph.items():
        keyword_score_max = max(keyword_score_max, value[0])
        keyword_score_min = min(keyword_score_min, value[0])
    for key, value in candidate_social_subgraph.items():
        time_score_max = max(time_score_max, value[1])
        time_score_min = min(time_score_min, value[1])

    denominator_keyword_score = keyword_score_min - keyword_score_max
    if denominator_keyword_score == 0:
        denominator_keyword_score = -keyword_score_max - eps

    denominator_time_score = time_score_min - time_score_max
    if denominator_time_score == 0:
        denominator_time_score = time_score_max - eps

    mean_keyword_score = sum(value[0] for key, value in candidate_social_subgraph.items())/len(candidate_social_subgraph.keys())
    mean_time_score = sum(value[1] for key, value in candidate_social_subgraph.items())/len(candidate_social_subgraph.keys())


    f_temp = 0
    for key, value in candidate_social_subgraph.items():
        f_keyword = (value[0] - keyword_score_max) / denominator_keyword_score
        f_time = ((value[1]) - time_score_max) / denominator_time_score
        f_temp = f_keyword + f_time
        if mean_keyword_score != 0 and mean_time_score != 0:
            dev = (value[1])/mean_time_score - value[0]/mean_keyword_score
        else:
            dev = (value[1]) - value[0]
        temp_dict.update({key: (round(f_temp, 8), round(dev, 8))})
    return list(dict(sorted(temp_dict.items(), key=lambda item: (item[1][0], item[1][1]))).keys())

def expanding_procedure(expanding_subgraph, candidate_node, expanding_social_subgraph, social_graph, T, t, vr, road_graph, uq, timetable, road_shortest_timetable):
    temp_neighbor = list()
    for i in social_graph.edge[candidate_node]:
        if i in expanding_subgraph.edge.keys():
            temp_neighbor.append(i)
            expanding_subgraph.edge[i].append(candidate_node)
    expanding_subgraph.edge.update({candidate_node: temp_neighbor})
    expanding_subgraph.keyword_score = compute_keyword_score_entire_community(expanding_subgraph, T, social_graph, uq)
    expanding_subgraph.time_score = compute_average_time(expanding_subgraph, road_graph, social_graph, t, vr, timetable, road_shortest_timetable)
    temp_subgraph = copy.deepcopy(expanding_subgraph)
    expanding_social_subgraph.update({candidate_node: temp_subgraph})

def pick_opt_subgraph_new(social_subgraph, expanding_social_subgraph, expanding_subgraph):

    temp_dict = dict()
    keyword_score_max = float('-inf')
    keyword_score_min = float('inf')
    time_score_max = float('-inf')
    time_score_min = float('inf')
    mean_keyword_score = 0
    mean_time_score = 0
    eps = 0.00000001

    for key, value in expanding_social_subgraph.items():
        keyword_score_max = max(keyword_score_max, value.keyword_score)
        keyword_score_min = min(keyword_score_min, value.keyword_score)

    for key, value in expanding_social_subgraph.items():
        time_score_max = max(time_score_max, value.time_score)
        time_score_min = min(time_score_min, value.time_score)

    mean_keyword_score = sum(value.keyword_score for key, value in expanding_social_subgraph.items())/len(expanding_social_subgraph.keys())
    mean_time_score = sum((value.time_score) for key, value in expanding_social_subgraph.items())/len(expanding_social_subgraph.keys())

    denominator_keyword_score = keyword_score_min - keyword_score_max
    if denominator_keyword_score == 0:
        denominator_keyword_score = -keyword_score_max - eps

    denominator_time_score = time_score_min - time_score_max
    if denominator_time_score == 0:
        denominator_time_score = time_score_max - eps

    f_temp = 0

    for key, value in expanding_social_subgraph.items():

        f_keyword = (value.keyword_score - keyword_score_max) / denominator_keyword_score
        f_time = ((value.time_score)- time_score_max) / denominator_time_score
        f_temp = f_keyword + f_time
        value.MD = f_temp
        if mean_keyword_score != 0 and mean_time_score != 0:
            dev = (value.time_score)/mean_time_score - value.keyword_score/mean_keyword_score
        else:
            dev = (value.time_score) - value.keyword_score

        temp_dict.update({key: (round(f_temp, 8), round(dev, 8))})
    previous_key = -1
    previous_value = float('inf')
    for key, value in temp_dict.items():
        if float(value[0]) < previous_value:
            previous_key = key
            previous_value = float(value[0])
        else:
            break
    temp_list = list(dict(sorted(temp_dict.items(), key=lambda item: (item[1][0], item[1][1]))).keys())

    return expanding_social_subgraph[previous_key]

def compute_subgraph_history_MD_new(subgraph_history, social_subgraph_opt):
    keyword_score_max = float('-inf')
    keyword_score_min = float('inf')
    time_score_max = float('-inf')
    time_score_min = float('inf')
    keyword_score_max = max(keyword_score_max, social_subgraph_opt.keyword_score)
    keyword_score_min = min(keyword_score_min, social_subgraph_opt.keyword_score)
    time_score_max = max(time_score_max, social_subgraph_opt.time_score)
    time_score_min = min(time_score_min, social_subgraph_opt.time_score)
    eps = 0.00000001
    for i in subgraph_history:
        keyword_score_max = max(keyword_score_max, i.keyword_score)
        keyword_score_min = min(keyword_score_min, i.keyword_score)
    for i in subgraph_history:
        time_score_max = max(time_score_max, i.time_score)
        time_score_min = min(time_score_min, i.time_score)
    keyword_score_denominator = keyword_score_min - keyword_score_max
    if keyword_score_denominator == 0:
        keyword_score_denominator = -keyword_score_max - eps

    time_score_denominator = time_score_min - time_score_max
    if time_score_denominator == 0:
        time_score_denominator = time_score_max = eps

    history_mini_MD = int(1e+8)
    f_keyword = (social_subgraph_opt.keyword_score - keyword_score_max) / keyword_score_denominator
    f_time = (social_subgraph_opt.time_score - time_score_max) / time_score_denominator
    social_subgraph_opt.MD = f_keyword + f_time

    for i in subgraph_history:
        f_keyword_h = (i.keyword_score - keyword_score_max) / keyword_score_denominator
        f_time_h = (i.time_score - time_score_max) / time_score_denominator
        MD_key = f_keyword_h + f_time_h
        i.MD = MD_key
        if MD_key < history_mini_MD:
            history_mini_MD = MD_key
            history_subgraph_opt = i
    return history_mini_MD, history_subgraph_opt


def greedy_method_flickr_NA_controller(uq, k, social_graph, road_graph):
    start_time = time.time()

    if uq not in social_graph.edge.keys():
        return 0, None, 0, None
    social_subgraph, flag = find_minimum_k_core_7(social_graph, k, uq, road_graph)
    initial_community = copy.deepcopy(social_subgraph)
    if flag == 0:
        return 0, social_subgraph, 0, social_subgraph

    start_community_size = len(social_subgraph.edge.keys())
    T = social_graph.keyword[uq]

    keyword_score = compute_keyword_score_entire_community(social_subgraph, T, social_graph, uq)
    social_subgraph.keyword_score = keyword_score

    t = 0
    vr = social_graph.location[uq]
    timetable = {}
    road_shortest_timetable = {}
    average_time = compute_average_time(social_subgraph, road_graph, social_graph, t, vr, timetable, road_shortest_timetable)
    social_subgraph.time_score = average_time

    neighbors_of_subgraph = set()
    subgraph_node = set(social_subgraph.edge.keys())
    temp_set = set()
    for key in social_subgraph.edge.keys():
        temp_set.update(social_graph.edge[key])
    neighbors_of_subgraph = temp_set - subgraph_node

    subgraph_history = list()
    candidate_neighbors = list()
    candidate_social_subgraph = dict()
    expanding_social_subgraph = dict()

    while True:
        candidate_neighbors.clear()
        candidate_social_subgraph.clear()
        for neighbor in neighbors_of_subgraph:
            if len(social_graph.edge[neighbor]) >= k:
                neighbor_link_num = len(set(social_graph.edge[neighbor]).intersection(set(social_subgraph.edge.keys())))
                if neighbor_link_num >= k:
                    non_dominance = judge_dominance(social_subgraph, neighbor, social_graph, T, road_graph, t, vr,
                                                    candidate_social_subgraph, uq, timetable, road_shortest_timetable)
                    if non_dominance == 1:
                        candidate_neighbors.append(neighbor)
                        if len(candidate_neighbors) >= 6:
                            break
                else:
                    continue
        if len(candidate_neighbors) == 0:
            break
        elif len(candidate_neighbors) > 1:
            candidate_neighbors = sort_candidate_neighbors_new(social_subgraph, candidate_social_subgraph)

        expanding_subgraph = copy.deepcopy(social_subgraph)
        expanding_social_subgraph.clear()
        for candidate_node in candidate_neighbors:
            expanding_procedure(expanding_subgraph, candidate_node, expanding_social_subgraph, social_graph, T, t, vr, road_graph, uq, timetable, road_shortest_timetable)
        if len(expanding_social_subgraph.keys()) == 1:
            social_subgraph_opt = expanding_subgraph
        else:
            social_subgraph_opt = pick_opt_subgraph_new(social_subgraph, expanding_social_subgraph, expanding_subgraph)
        if len(subgraph_history) < 1:
            subgraph_history.append(social_subgraph_opt)
        else:
            history_mini_MD, history_subgraph_opt = compute_subgraph_history_MD_new(subgraph_history, social_subgraph_opt)
            if social_subgraph_opt.MD < history_mini_MD:
                subgraph_history.append(social_subgraph_opt)
            else:
                social_subgraph = history_subgraph_opt
                break

        social_subgraph = social_subgraph_opt
        already_in_subgraph_node = neighbors_of_subgraph.intersection(set(social_subgraph.edge.keys()))
        for i in already_in_subgraph_node:
            neighbors_of_subgraph.update(set(social_graph.edge[i]))
        neighbors_of_subgraph = neighbors_of_subgraph - set(social_subgraph.edge.keys())
    if len(social_subgraph.edge.keys()) >= start_community_size:
        return 1, social_subgraph, time.time() - start_time, initial_community
    else:
        return 0, social_subgraph, 0, social_subgraph