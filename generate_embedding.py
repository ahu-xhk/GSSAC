import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_chat_messages(prompt_text):
    
    with open('config.json') as config_file:
        config = json.load(config_file)

    gpt_model = config['gpt_model']

    response = openai.ChatCompletion.create(
        model = gpt_model,
        messages = prompt_text,
        max_tokens = 4096,
        n = 1,
        stop = None,
        temperature = 0
    )
    return response.choices[0].message.content

def generate_embeddings(text_list):
    with open('config.json') as config_file:
        config = json.load(config_file)

    embedding_model = config['embedding_model']

    combined_text = " ".join(text_list)
    response = openai.Embedding.create(
        model= embedding_model, 
        input= combined_text
    )
    return response['data'][0]['embedding']


def cosine_similarity_for_vec(embedding_a, embedding_b):
    A = np.array(embedding_a).reshape(1, -1)  
    B = np.array(embedding_b).reshape(1, -1)
    similarity = cosine_similarity(A, B)[0][0]
    return round(similarity, 8)

def cosine_similarity_for_mat(matrix_a, matrix_b):
    matrix_1 = np.array(matrix_a)
    matrix_2 = np.array(matrix_b)

    cos_sim_matrix = cosine_similarity(matrix_1, matrix_2)

    max_sim_idx = np.unravel_index(np.argmax(cos_sim_matrix), cos_sim_matrix.shape)
    max_similarity = cos_sim_matrix[max_sim_idx]
    vector1 = matrix_1[max_sim_idx[0]]
    vector2 = matrix_2[max_sim_idx[1]]

    return round(max_similarity, 8)


def get_embedding_file():
    with open('config.json') as config_file:
        config = json.load(config_file)

    openai.api_base = config['openai_api_base']
    openai.api_key = config['openai_api_key']
    system_prompt_file = config['system_prompt']
    user_prompt_file = "../03_prompts/" + config['experiment_dataset'] + "_user_prompts.txt"
    user_embeddings_file = "../04_vectors/" + config['experiment_dataset'] + "_embeddings.json"


    with open(system_prompt_file, 'r', encoding='utf-8') as file:
        system_prompt = file.read()


    with open(user_prompt_file, 'r', encoding='utf-8') as file:
        user_prompt = file.read()


    prompt_text = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}]

    response_content = get_chat_messages(prompt_text)


    print("ChatGPT：", response_content)


    user_data = json.loads(response_content)




    user_embeddings = {}

    for user in user_data:
        user_id = user['user_id']
        combined_keywords = user['filtered_keywords'] + user['generated_keywords']
        embeddings = generate_embeddings(combined_keywords)
        user_embeddings[user_id] = embeddings
    for user_id, embedding in user_embeddings.items():
        print(f"User ID: {user_id}, Embedding Length: {len(embedding)}")
    try:
        with open(user_embeddings_file, 'r') as f:
            user_embeddings_load = json.load(f)  
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        user_embeddings_load = {}

    user_embeddings_load.update(user_embeddings)

    keys = list(user_embeddings_load.keys())
    with open(user_embeddings_file, 'w') as f:
        f.write('{\n')
        for i, key in enumerate(keys):
            f.write(f'    "{key}": ')
            f.write(json.dumps(user_embeddings_load[key]))
            if i < len(keys) - 1:
                f.write(',\n')
            else:
                f.write('\n')

        f.write('}\n')

def compute_semantic_similarity(node1, node2):
    with open('config.json') as config_file:
        config = json.load(config_file)

    user_embeddings_file = "../04_vectors/" + config['experiment_dataset'] + "_embeddings.json"

    try:
        with open(user_embeddings_file, 'r') as f:
            data = json.load(f)  
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data = {}

    user_id_1 = str(node1)
    user_id_2 = str(node2)

    embedding_a_origin = data[user_id_1]
    embedding_b_origin = data[user_id_2]

    similarity_origin = cosine_similarity_for_vec(embedding_a_origin, embedding_b_origin)
    return similarity_origin

def generate_user_prompts(user_keywords_dict):
    with open('config.json') as config_file:
        config = json.load(config_file)
    user_prompt_file = "../03_prompts/" + config['experiment_dataset'] + "_user_prompts.txt"

    count = 0
    with open(user_prompt_file, "w", encoding="utf-8") as file:
        length = len(user_keywords_dict)
        file.write("[\n")
        for key, value in user_keywords_dict.items():
            count += 1
            keywords = '\", \"'.join(map(str, value))
            if count != length:
                file.write("  {\"user_id\": \"" + str(key) + "\", \"keywords\": [\"" + keywords + '\"]},\n')
            else:
                file.write("  {\"user_id\": \"" + str(key) + "\", \"keywords\": [\"" + keywords + '\"]}\n')
                file.write("]")
                break

def get_embedding_for_community_keyword_set(community_keywords):
    with open('config.json') as config_file:
        config = json.load(config_file)

    openai.api_base = config['openai_api_base']
    openai.api_key = config['openai_api_key']
    system_prompt_file = config['system_prompt']

    with open(system_prompt_file, 'r', encoding='utf-8') as file:
        system_prompt = file.read()
    temp_dict = {"user_id" : "0", "keywords": list(community_keywords)}
    json_data = json.dumps(temp_dict, ensure_ascii=False)
    user_prompt = str(json_data)
    print(user_prompt)

    prompt_text = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}]

    response_content = get_chat_messages(prompt_text)

    print("ChatGPT：", response_content)

    user_data = json.loads(response_content)

    combined_keywords = user_data[0]['filtered_keywords'] + user_data[0]['generated_keywords']

    return generate_embeddings(combined_keywords)

def get_embedding_for_query_node_keyword_set(query_node, social_graph):
    with open('config.json') as config_file:
        config = json.load(config_file)

    user_embeddings_file = "../04_vectors/" + config['experiment_dataset'] + "_embeddings.json"

    try:
        with open(user_embeddings_file, 'r') as f:
            embedding_dict = json.load(f)  
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        embedding_dict = {}


    if str(query_node) in embedding_dict.keys():
        return embedding_dict[str(query_node)]

    with open('config.json') as config_file:
        config = json.load(config_file)

    openai.api_base = config['openai_api_base']
    openai.api_key = config['openai_api_key']
    system_prompt_file = config['system_prompt']

    with open(system_prompt_file, 'r', encoding='utf-8') as file:
        system_prompt = file.read()

    temp_dict = {"user_id": str(query_node), "keywords": list(social_graph.keyword[query_node])}
    json_data = json.dumps(temp_dict, ensure_ascii=False)
  
    user_prompt = str(json_data)
    print(user_prompt)

    prompt_text = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}]

    response_content = get_chat_messages(prompt_text)

    print("ChatGPT：", response_content)

    user_data = json.loads(response_content)


    user_embeddings = {}

    user_id = user_data[0]['user_id']
    combined_keywords = user_data[0]['filtered_keywords'] + user_data[0]['generated_keywords']
    embeddings = generate_embeddings(combined_keywords)
    user_embeddings[user_id] = embeddings

    for user_id, embedding in user_embeddings.items():
        print(f"User ID: {user_id}, Embedding Length: {len(embedding)}")

    try:
        with open(user_embeddings_file, 'r') as f:
            user_embeddings_load = json.load(f)  
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        user_embeddings_load = {}

    user_embeddings_load.update(user_embeddings)

    keys = list(user_embeddings_load.keys())
    with open(user_embeddings_file, 'w') as f:
        f.write('{\n')

        for i, key in enumerate(keys):
            f.write(f'    "{key}": ')
            f.write(json.dumps(user_embeddings_load[key]))
            if i < len(keys) - 1:
                f.write(',\n')
            else:
                f.write('\n')

        f.write('}\n')

    return embeddings
