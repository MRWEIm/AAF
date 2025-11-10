import torch
import itertools
import utils
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import json
from tqdm import tqdm
from probe import MLP_Regressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from qwk_cal import qwk_calculation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor

def split_data_2_8(data, label):
    total_samples = data.shape[0]
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=g)

    # 获取训练集和测试集
    train_data = data[indices[:train_size]]
    test_data = data[indices[train_size:]]

    train_label = label[indices[:train_size]]
    test_label = label[indices[train_size:]]

    return train_data, test_data, train_label, test_label

def split_data_cross_prompt(data, label, prompt_id, without_7=False):
    if without_7:
        train_data_list = [d for i, d in enumerate(data) if isinstance(d, torch.Tensor) and i != (prompt_id - 1) and i != 6]
        train_label_list = [l for i, l in enumerate(label) if isinstance(l, torch.Tensor) and i != (prompt_id - 1) and i != 6]
    else:
        train_data_list = [d for i, d in enumerate(data) if isinstance(d, torch.Tensor) and i != (prompt_id - 1)]
        train_label_list = [l for i, l in enumerate(label) if isinstance(l, torch.Tensor) and i != (prompt_id - 1)]

    train_data = torch.cat(train_data_list, dim=0)
    train_label = torch.cat(train_label_list, dim=0)
    test_data = data[prompt_id - 1]
    test_label = label[prompt_id - 1]
    return train_data, test_data, train_label, test_label


class head_info():
    def __init__(self, layer, head, pred_score):
        self.layer = layer
        self.head = head
        self.pred_score = pred_score

    def renew(self, layer, head, pred_score):
        self.layer = layer
        self.head = head
        self.pred_score = pred_score

    def dict_form(self):
        return {
            'layer': self.layer,
            'head': self.head,
            'pred_score': round(self.pred_score, 4)    
        }

max_values = [[12,  6,  6,  6,  6,  6, -1, -1, -1],
              [ 6,  6,  6,  6,  6,  6, -1, -1, -1],
              [ 3,  3, -1, -1, -1, -1,  3,  3,  3],
              [ 3,  3, -1, -1, -1, -1,  3,  3,  3],
              [ 4,  4, -1, -1, -1, -1,  4,  4,  4],
              [ 4,  4, -1, -1, -1, -1,  4,  4,  4],
              [30,  6,  6, -1, -1,  6, -1, -1, -1],
              [60, 12, 12, 12, 12, 12, -1, -1, -1]]

min_values = [[2, 1,  1,  1,  1,  1, -1, -1, -1],
              [1, 1,  1,  1,  1,  1, -1, -1, -1],
              [0, 0, -1, -1, -1, -1,  0,  0,  0],
              [0, 0, -1, -1, -1, -1,  0,  0,  0],
              [0, 0, -1, -1, -1, -1,  0,  0,  0],
              [0, 0, -1, -1, -1, -1,  0,  0,  0],
              [0, 0,  0, -1, -1,  0, -1, -1, -1],
              [0, 2,  2,  2,  2,  2, -1, -1, -1]]

def regression(prompt_id, trait, probe_type='Linear', draw_heatmap=False, prompt_type='all', train_type='cross_prompt', without_7=False, model_name='Qwen3-8B', alpha=1):
    assert train_type in ['cross_prompt', 'single_prompt']
    assert probe_type in ['Linear', 'MLP']
    assert prompt_type in ['all', 'wo_p', 'wo_i', 'only_e']

    index = utils.load_json_file('./AES/Json/Index.json')
    if prompt_type in ['wo_i', 'only_e']:
        head_wise_activations = [torch.load(f'./AES/ASAP/activations/{model_name}/{model_name}_Prompt_{ii}_None_{prompt_type}.pt') 
                                if trait in index[f'Prompt_{ii}'] else 0
                                for ii in range(1, 9)]
    else:
        head_wise_activations = [torch.load(f'./AES/ASAP/activations/{model_name}/{model_name}_Prompt_{ii}_{trait}_{prompt_type}.pt') 
                                if trait in index[f'Prompt_{ii}'] else 0
                                for ii in range(1, 9)]
    score = [torch.load(f'./AES/ASAP/score/Prompt_{ii}.pt')[:, index[f'Prompt_{ii}'][trait]]
             if trait in index[f'Prompt_{ii}'] else 0
             for ii in range(1, 9)]
    
    if train_type == 'cross_prompt':
        train_data, test_data, train_label, test_label = split_data_cross_prompt(data=head_wise_activations, label=score, prompt_id=prompt_id, without_7=without_7)
    elif train_type == 'single_prompt':
        train_data, test_data, train_label, test_label = split_data_2_8(data=head_wise_activations[prompt_id-1], label=score[prompt_id-1])
    diff = max_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]] - min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    bias = min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    test_label = (test_label * diff + bias).int()
    

    model_score, model_head_info = [], []
    probe_dict = {}
    min_score = head_info(0, 0, 1)
    max_score = head_info(0, 0, 0)
    for layer in tqdm(range(train_data.shape[1])):
        layer_score = []
        for head in range(train_data.shape[2]):
            if probe_type == 'Linear':
                probe = Ridge(alpha=alpha, fit_intercept=False)
            elif probe_type == 'MLP':
                probe = MLP_Regressor(input_size=128, hidden_size=256, output_size=1, epochs=300, patience=3, learning_rate=1e-3, weight_decay=0.01, batch_size=4096 if train_type == 'cross_prompt' else 512)
            probe.fit(train_data[:, layer, head, :], train_label)
            probe_dict[f'L{layer}-H{head}'] = probe

            y_pred = probe.predict(test_data[:, layer, head, :])
            y_pred = np.clip(y_pred, 0, 1)      
            y_pred = np.rint(diff * y_pred + bias).astype(int) 
            y_pred = torch.tensor(y_pred, dtype=torch.int32)
            pred_score = qwk_calculation(test_label, y_pred)
            layer_score.append(pred_score)
            model_head_info.append(head_info(layer, head, pred_score))   

            if pred_score > max_score.pred_score:
                max_score.renew(layer, head, pred_score)
            if pred_score < min_score.pred_score:
                min_score.renew(layer, head, pred_score)

        model_score.append(layer_score)
    model_score = np.array(model_score)

    print(f'Prompt_id: {prompt_id} trait: {trait} Layer: {min_score.layer}, Head: {min_score.head} Min Pred Score: {min_score.pred_score:.4f}')
    print(f'Prompt_id: {prompt_id} trait: {trait} Layer: {max_score.layer}, Head: {max_score.head} Max Pred Score: {max_score.pred_score:.4f}')

    if draw_heatmap:
        model_score = np.sort(model_score)[:, ::-1]
        model_score = model_score[::-1]
        sns.heatmap(model_score, cmap='viridis_r', vmin=0, vmax=0.85 if train_type == 'single_prompt' else 0.7, xticklabels=False, yticklabels=True)
        plt.xlabel('Head(Sorted)', fontsize=30)
        plt.ylabel('Layer', fontsize=30)
        step = 2
        ticks = np.arange(32)
        labels = list(reversed(range(32)))

        plt.yticks(ticks=ticks[::step] + 1.5, labels=[labels[i] for i in range(0, len(labels), step)], rotation=0, fontsize=18)
        plt.savefig(f'./AES/ASAP/plot/{model_name}_P{prompt_id}_{trait}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    return max_score, model_head_info, probe_dict


def spec_head_regression(prompt_id, trait, layer, head, probe_type='Linear'):
    index = utils.load_json_file('./AES/Json/Index.json')
    head_wise_activations = [torch.load(f'./AES/ASAP/activations/Prompt_{ii}_{trait}_head_wise.pt') 
                             if trait in index[f'Prompt_{ii}'] else 0
                             for ii in range(1, 9)]
    score = [torch.load(f'./AES/ASAP/score/Prompt_{ii}.pt')[:, index[f'Prompt_{ii}'][trait]]
             if trait in index[f'Prompt_{ii}'] else 0
             for ii in range(1, 9)]

    train_data, test_data, train_label, test_label = split_data_cross_prompt(data=head_wise_activations, label=score, prompt_id=prompt_id)
    diff = max_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]] - min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    bias = min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    test_label = (test_label * diff + bias).int()

    if probe_type == 'Linear':
        probe = Ridge(alpha=1, fit_intercept=False)
    elif probe_type == 'NonLinear':
        probe = MLPRegressor(hidden_layer_sizes=(128, ), activation='relu', max_iter=500, random_state=42)
    probe.fit(train_data[:, layer, head, :], train_label)
    y_pred = probe.predict(test_data[:, layer, head, :])
    y_pred = np.clip(y_pred, 0, 1)      
    y_pred = np.rint(diff * y_pred + bias).astype(int) 
    y_pred = torch.tensor(y_pred, dtype=torch.int32)
    score = qwk_calculation(test_label, y_pred)

    return score


def visualize_activations(prompt_id, trait, layer, head, type='t-SNE'):
    """
    对特定head的激活向量进行可视化。

    Args:
        prompt_id (int): 提示ID。
        trait (str): 特征名称。
        layer (int): 层数。
        head (int): 头数。
        type (str, optional): 可选的可视化类型，可以是't-SNE'或'PCA'。默认为't-SNE'。

    Returns:
        None
    """
    head_wise_activations = torch.load(f'./AES/ASAP/activations/Prompt_{prompt_id}_{trait}_all.pt')
    activations = head_wise_activations[:, layer, head, :].detach().cpu().numpy()
    index = utils.load_json_file('./AES/Json/Index.json')
    score = torch.load(f'./AES/ASAP/score/Prompt_{prompt_id}.pt')[:, index[f'Prompt_{prompt_id}'][trait]]
    diff = max_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]] - min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    bias = min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    score = (score * diff + bias).clone().detach().to(torch.int32).cpu().numpy()

    if type == 'PCA':
        pca_2d = PCA(n_components=2)
        activations_2d = pca_2d.fit_transform(activations)
    elif type == 't-SNE':
        random = 42
        tsne_2d = TSNE(n_components=2, random_state=random, perplexity=30, max_iter=1000)
        activations_2d = tsne_2d.fit_transform(activations)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(activations_2d[:, 0], activations_2d[:, 1],
                          c=score, cmap='viridis', s=20, alpha=0.8)
    # plt.colorbar(scatter, label='Score')
    # plt.title("t-SNE of 128-D tensor features")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    plt.grid(False)
    plt.savefig(f'./AES/ASAP/plot/P_{prompt_id}_{trait}_L_{layer}_H_{head}_{type}_2d.png')
    plt.close()


def get_spec_head_avg_score(prompt_id_list, trait_list, layer, head):
    """
    计算指定层和头在指定特征空间下的平均得分。

    Args:
        prompt_id (int): 提示ID。
        trait (str): 特征名称。
        layer (int): 层数。
        head (int): 头数。

    Returns:
        None
    """
    score_list = []
    for prompt_id, trait in itertools.product(prompt_id_list, trait_list):
        if not os.path.exists(f'./AES/ASAP/activations/Prompt_{prompt_id}_{trait}_head_wise.pt'):
            continue
        print(f'Processing Prompt_{prompt_id} {trait}')
        score = spec_head_regression(prompt_id=prompt_id, trait=trait, layer=layer, head=head)
        score_list.append(score)
    
    avg = sum(score_list) / len(score_list)
    print(avg)  



def process_one_pair(args):
    trait, prompt_id = args
    print(f'Processing Prompt_{prompt_id} {trait}')
    try:
        max_score, model_head_info, _ = regression(
            prompt_id=prompt_id,
            trait=trait,
            probe_type='MLP',
            train_type='cross_prompt',
            prompt_type='all',
            draw_heatmap=True,
            model_name='Deepseek',
        )
        return (trait, prompt_id, model_head_info)
    except Exception as e:
        print(f"Error in Prompt_{prompt_id} {trait}: {e}")
        return None

def MLP_regression(prompt_id_list, trait_list):
    index = utils.load_json_file('./AES/Json/Index.json')
    pairs_to_process = []
    for trait, prompt_id in itertools.product(trait_list, prompt_id_list):
        if trait not in index[f'Prompt_{prompt_id}']:
            continue
        pairs_to_process.append((trait, prompt_id))

    with ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(process_one_pair, pairs_to_process):
            if result is None:
                continue
            trait, prompt_id, _ = result


def process_one_pair_linear(args):
    trait, prompt_id, alpha, model_name = args
    print(f'Processing Prompt_{prompt_id} {trait}')
    try:
        max_score, model_head_info, _ = regression(
            prompt_id=prompt_id,
            trait=trait,
            probe_type='Linear',
            train_type='cross_prompt',
            prompt_type='all',
            draw_heatmap=True,
            model_name=model_name,
            without_7=True,
            alpha=alpha
        )
        return (trait, prompt_id, model_head_info)
    except Exception as e:
        print(f"Error in Prompt_{prompt_id} {trait}: {e}")
        return None

def Linear_regression(prompt_id_list, trait_list, alpha, model_name):
    index = utils.load_json_file('./AES/Json/Index.json')
    pairs_to_process = []
    for trait, prompt_id in itertools.product(trait_list, prompt_id_list):
        if trait not in index[f'Prompt_{prompt_id}']:
            continue
        if prompt_id == 7:
            continue
        # if (trait != 'Narrativity' or prompt_id != 3) and (trait != 'Content' or prompt_id != 5):
        #     continue
        pairs_to_process.append((trait, prompt_id, alpha, model_name))

    with ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(process_one_pair_linear, pairs_to_process):
            if result is None:
                continue
            trait, prompt_id, _ = result


if __name__ == '__main__':
    random.seed(42)
    prompt_id_list = [1, 2, 3, 4, 5, 6, 7, 8]
    trait_list = ['Holistic', 'Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Prompt Adherence', 'Language', 'Narrativity']
    alpha_list = [0, 0.01, 0.1, 1, 10]
    # prompt_id_list = [3]
    # trait_list = ['Holistic', 'Content', 'Organization', 'Conventions']
    # alpha_list = [0.01]
    prompt_type = 'all'
    probe_type = 'Linear'
    train_type = 'cross_prompt'
    model_name = 'Llama-2-7b-chat-hf'


    if probe_type == 'MLP':
        MLP_regression(prompt_id_list, trait_list)
    else:
        prompt_model_head_info = {}
        index = utils.load_json_file('./AES/Json/Index.json')
        for alpha in alpha_list:
            for trait, prompt_id in itertools.product(trait_list, prompt_id_list):
                if trait not in index[f'Prompt_{prompt_id}']:
                    continue
                print(f'Processing Prompt_{prompt_id} {trait} {prompt_type}')

                max_score, model_head_info, _ = regression(prompt_id=prompt_id, trait=trait, probe_type=probe_type, train_type=train_type, prompt_type=prompt_type, draw_heatmap=False, model_name=model_name, alpha=alpha)
                # visualize_activations(prompt_id=prompt_id, trait=trait, layer=max_score.layer, head=max_score.head)

                if trait not in prompt_model_head_info:
                    prompt_model_head_info[trait] = {}
                prompt_model_head_info[trait][f'Prompt_{prompt_id}'] = [s.dict_form() for s in model_head_info]
                print('='*50)

            with open(f'./AES/ASAP/head_info/{model_name}_{train_type}_{probe_type}_model_head_info.json', 'w') as f:
                json.dump(prompt_model_head_info, f, indent=4)
