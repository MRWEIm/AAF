import csv
import json
import pickle

column_map = {
    1: [3, 4, 5, 6, 7, 8],  # indices for essay_set == 1
    2: [3, 4, 5, 6, 7, 8],  # indices for essay_set == 2
    3: [3, 4, 9, 10, 11],  # indices for essay_set == 3-7
    4: [3, 4, 9, 10, 11],  # same as 3
    5: [3, 4, 9, 10, 11],  # same as 3
    6: [3, 4, 9, 10, 11],  # same as 3
    7: [3, 4, 5, 8],  # same as 3
    8: [3, 4, 5, 6, 7, 8],  # indices for essay_set == 8
}
def load_essay_data(file_path):
    """
    get essay text and prompt text

    Args:
        file_path: text file path

    Return:
        essay_text: [prompt_id][essay][text, total_score, trait_score]
    """
    essay_texts = [[] for _ in range(8)]
    essay_scores = [[] for _ in range(8)]
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            tokens = line.strip().split('\t')
            essay_id = tokens[0]
            essay_set = int(tokens[1])      # get prompt id
            content = tokens[2].strip()     # get essay text

            column_indices = column_map.get(essay_set)
            if column_indices:
                essay_texts[essay_set - 1].append(content)

                essay_score = [int(float(tokens[i])) for i in column_indices]
                essay_scores[essay_set - 1].append(essay_score)

    return essay_texts, essay_scores

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file 

def load_prompt(file_path, prompt_id=None):
    file = load_json_file(file_path)
    if prompt_id:
        return file[f'Prompt_{prompt_id}']
    return file

def open_pk(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def open_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def open_tsv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
    return data



if __name__ == '__main__':
    print(1)