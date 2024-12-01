import sys
import greedy
import os
import json

def list_files_in_directory(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file = filename.strip().split(".")
            files.append(int(file[0]))
    return files

def load_query_nodes(directory):
    with open(directory, 'r') as file:
        content = file.read()
        number_strings = content.split(',')
        numbers = [int(num.strip()) for num in number_strings]
    return numbers


class Social_Graph():
    def __init__(self):
        self.edge = {}
        self.location = {}
        self.keyword = {}
        self.keyword_score = int()
        self.time_score = int()
        self.MD = int()


class Road_Graph():
    def __init__(self):
        self.edge = {}
        self.time = {}
        self.coordinate = {}

def construct_networks(f1_social_edge, f2_social_location_keyword, f3_road_edge, f4_road_time, f5_road_coordinate):
    social_graph = Social_Graph()
    road_graph = Road_Graph()
    with open(f1_social_edge, "r") as f1:
        for line in f1:
            curline = line.strip().split(" ")
            curline = [int(x) for x in curline]
            if len(curline) > 1:
                temp = []
                for i in range(1, len(curline)):
                    temp.append(curline[i])
                social_graph.edge.update({curline[0]: temp})
            else:
                social_graph.edge.update({curline[0]: []})
    with open(f2_social_location_keyword, "r") as f2:
        for line in f2:
            curline = line.strip().split(",")
            social_graph.location.update({int(curline[0]): int(curline[1])})
            keyword_set = set(curline[2].strip().split(" "))
            social_graph.keyword.update({int(curline[0]): keyword_set})
    with open(f3_road_edge, "r") as f3:
        for line in f3:
            curline = line.strip().split(" ")
            curline = [int(x) for x in curline]
            if len(curline) > 1:
                temp = []
                for i in range(1, len(curline)):
                    temp.append(curline[i])
                road_graph.edge.update({curline[0]: temp})
            else:
                road_graph.edge.update({curline[0]: []})
    with open(f4_road_time, "r") as f4:
        for line in f4:
            curline = line.strip().split(" ")
            curline = [int(x) for x in curline]
            curtuple = (curline[0], curline[1])
            curlist = curline[2:]
            road_graph.time.update({curtuple: curlist})
    with open(f5_road_coordinate, "r") as f5:
        for line in f5:
            curline = line.strip().split(" ")
            curline = [int(curline[0])] + [float(x) for x in curline[1:]]
            road_graph.coordinate.update({curline[0]: curline[1:]})
    return social_graph, road_graph

if __name__ == "__main__":

    with open('config.json') as config_file:
        config = json.load(config_file)

    index = config['experiment_dataset'].rfind('_')

    social_dataset = config['experiment_dataset'][:index]  
    road_dataset = config['experiment_dataset'][index + 1:]  

    social_graph, road_graph = construct_networks(f1, f2, f3, f4, f5)

    count = 0
    common_query_nodes = load_query_nodes("../02_query_nodes/" + config['experiment_dataset'] + "/common.txt")

    forACQ_query_nodes = load_query_nodes("../02_query_nodes/" + config['experiment_dataset'] + "/forACQ.txt")
    ACQ_query_nodes = list(set(common_query_nodes).intersection(set(forACQ_query_nodes)))

    list_file = list_files_in_directory("../05_results/" + config['experiment_dataset'])
    nodes = set(ACQ_query_nodes) - set(list_file)
    print(list(nodes))
    for uq in nodes:
        print(uq)
        k = int(config['k'])
        flag, community, run_time, initial_community = greedy_method_flickr_NA.greedy_method_flickr_NA_controller(uq, k, social_graph, road_graph)
        if flag == 1:
            count += 1
            with open("../05_results/" + config['experiment_dataset'] + "/" + str(uq) + ".txt", "w") as file:
                file.write("the_query_node_is：" + str(uq) + "\n")
                for key, value in community.edge.items():
                    neighbor_str = ' '.join(map(str, value))
                    file.write(str(key) + ' ' + neighbor_str + '\n')
                file.write(str(run_time) + 's')
