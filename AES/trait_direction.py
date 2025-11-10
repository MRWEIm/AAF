import torch
import utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def cal_direction(prompt_id, trait, layer, head, model_name):
    index = utils.load_json_file('./AES/Json/Index.json')
    spec_activations = torch.load(f'./AES/ASAP/activations/{model_name}/{model_name}_Prompt_{prompt_id}_{trait}_all.pt')[:, layer, head, :]
    spec_score = torch.load(f'./AES/ASAP/score/Prompt_{prompt_id}.pt')[:, index[f'Prompt_{prompt_id}'][trait]]
    assert isinstance(spec_activations, torch.Tensor) and isinstance(spec_activations, torch.Tensor) and spec_activations.shape[0] == spec_score.shape[0]

    diff = max_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]] - min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    bias = min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    spec_score = (spec_score * diff + bias).int()

    score_set = sorted(set(spec_score.tolist()))
    grouped_activations = [spec_activations[spec_score == score].mean(dim=0) for score in score_set]

    direction = []
    for ii in range(len(grouped_activations)):
        for jj in range(ii+1, len(grouped_activations)):
            direction.append(grouped_activations[jj] - grouped_activations[ii])
    direction = torch.stack(direction).mean(dim=0)
    direction = direction / direction.norm(p=2)
    return direction
    

def compare_direction(fixed_item, varying='trait', layer=0, head=0, model_name='', draw_heatmap=False):
    '''
    对向量方向进行对比

    Args:
        fixed_item (int or str): 固定项，如trait或prompt_id
        varying (str): 变化项，如trait或prompt_id或head
                       'prompt': 固定trait和Top 1的head，遍历8个prompt_id  
                       'trait': 固定prompt_id和Top 1的head，遍历9个trait    
                       'head': 固定prompt_id和trait，遍历Top 32的head
        method (str): 方向计算方法，可选'chain'或'stage'
                      
    Returns:
        None
    '''
    assert varying in ['trait', 'prompt']
    prompt_id_list = [1, 2, 3, 4, 5, 6, 7, 8]
    trait_list = ['Holistic', 'Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Prompt Adherence', 'Language', 'Narrativity']
    label_name = {'Holistic': 'Holistic', 'Content': 'Content', 'Organization': 'Org', 'Word Choice': 'WC',
                  'Sentence Fluency': 'SF', 'Conventions': 'Conv', 'Prompt Adherence': 'PA', 'Language': 'Lang', 'Narrativity': 'Nar'}
    index = utils.load_json_file('./AES/Json/Index.json')
    iter_list = trait_list if varying == 'trait' else (prompt_id_list if varying == 'prompt' else range(32))

    direction_dict = {}
    labels = []
    for iter_item in iter_list:
        if varying == 'trait':
            prompt_id = fixed_item
            trait = iter_item
        elif varying == 'prompt':
            prompt_id = iter_item
            trait = fixed_item

        if trait not in index[f'Prompt_{prompt_id}']:
            continue
        # labels.append(f'{label_name[iter_item]}')
        labels.append(f'P{iter_item}')
        direction = cal_direction(prompt_id=prompt_id, trait=trait, layer=layer, head=head, model_name=model_name)
        direction_dict[iter_item] = direction

    directions = torch.stack([d.float() for d in direction_dict.values()])
    directions = F.normalize(directions, p=2, dim=1)
    sim_matrix = directions @ directions.T
    # for ii in sim_matrix:
    #     print(ii)

    if draw_heatmap:
        plt.figure(figsize=(6, 6))
        sns.heatmap(sim_matrix.numpy(), cmap='Blues', vmin=0, vmax=1.0, annot=True, fmt='.2f', annot_kws={"size": 14})
        plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels)
        plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=90)
        plt.tight_layout()
        plt.savefig(f'./AES/ASAP/plot/{model_name}_{fixed_item}_direction_cos_sim.png')
        plt.close()

    sim_matrix = sim_matrix.numpy()
    sim_matrix = np.tril(sim_matrix, k=-1)
    sim_matrix = sim_matrix[sim_matrix != 0]
    return sim_matrix




def get_best_head_for_trait(prompt_model_head_info, trait):
    """
    获取head在不同prompt的平均分数。

    Args:
        prompt_sorted_model_head_info (list(list)): 提示排序后的模型头信息列表。

    Returns:
        sorted_indices (list): 排序后的索引列表。
        sorted_scores (list): 排序后的平均排名列表。
    """

    score = [0] * 36 * 32
    for prompt_head_info in prompt_model_head_info[trait].values():
        for head_info in prompt_head_info:
            score[head_info['layer'] * 32 + head_info['head']] += head_info['pred_score']
    score = [r / len(prompt_model_head_info[trait]) for r in score]
    sorted_indices = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    sorted_score = [score[i] for i in sorted_indices]
    return sorted_indices, sorted_score


def get_best_head_for_prompt(prompt_model_head_info, prompt_id):
    score = [0] * 36 * 32
    count = 0
    for trait_head_info in prompt_model_head_info.values():
        if f'Prompt_{prompt_id}' not in trait_head_info.keys():
            continue
        for head_info in trait_head_info[f'Prompt_{prompt_id}']:
            score[head_info['layer'] * 32 + head_info['head']] += head_info['pred_score']
        count += 1
    score = [r / max(count, 1) for r in score]
    sorted_indices = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    sorted_score = [score[i] for i in sorted_indices]
    return sorted_indices, sorted_score
    

def get_best_head(type='trait', model_name='Llama-2-7b-chat-hf'):
    assert type in ['trait', 'prompt']

    trait_list = ['Holistic', 'Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Prompt Adherence', 'Language', 'Narrativity']
    prompt_id_list = [i for i in range(1, 9)]
    iter_list = trait_list if type == 'trait' else prompt_id_list
    
    head_info = utils.load_json_file(f'{model_name}_cross_prompt_Linear_model_head_info.json')
    for iter in iter_list:
        if type == 'trait':
            sorted_indices, sorted_scores = get_best_head_for_trait(head_info, iter)
        elif type == 'prompt':
            sorted_indices, sorted_scores = get_best_head_for_prompt(head_info, iter)
        top_index, top_score = sorted_indices[0], sorted_scores[0]
        print(f'{iter} 的 Top 1 探针位置: L-{top_index // 32}-H-{top_index % 32}，分数为：{top_score:.4f}')


def lambda_effect():
    data = {
        'Llama-2-7b-chat-hf': {
            'Prompt': [0.6514, 0.6522, 0.6521, 0.6412, 0.6058],
            'Trait': [0.6503, 0.6511, 0.6516, 0.6403, 0.6015], 
        },
        'DeepSeek-R1-Distill-Llama-8B': {
            'Prompt': [0.6359, 0.6313, 0.6237, 0.6032, 0.5451],
            'Trait': [0.6339, 0.6297, 0.6201, 0.5992, 0.5376], 
        },
        'Qwen3-8B': {
            'Prompt': [0.6554, 0.6547, 0.6553, 0.6467, 0.6272],
            'Trait': [0.6537, 0.6530, 0.6528, 0.6448, 0.6232],
        },
    }
    x = ['0', '0.01', '0.1', '1', '10']

    plt.figure(figsize=(6, 4))
    for model_name, values in data.items():
        plt.plot(x, values['Prompt'], marker='o', label=model_name)
    plt.xlabel('λ', fontsize=12)
    plt.ylabel('Average QWK Score for Prompts', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'./AES/ASAP/plot/lambda_effect_prompt.png')

    plt.figure(figsize=(6, 4))
    for model_name, values in data.items():
        plt.plot(x, values['Trait'], marker='o', label=model_name)
    plt.xlabel('λ', fontsize=12)
    plt.ylabel('Average QWK Score for Traits', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'./AES/ASAP/plot/lambda_effect_trait.png')


if __name__ == '__main__':
    # get_best_head(type='prompt', model_name='DeepSeek-R1-Distill-Llama-8B')
    # lambda_effect()

    prompt_id, trait = 3, 'Content'
    layer, head = 15, 1

    model_name = ['Llama-2-7b-chat-hf', 'DeepSeek-R1-Distill-Llama-8B', 'Qwen3-8B']

    trait_list = ['Holistic', 'Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Prompt Adherence', 'Language', 'Narrativity']
    trait_info_list = {'Llama-2-7b-chat-hf': [(13, 12), (15, 1), (9, 2), (15, 4), (18, 15), (26, 27), (13, 12), (11, 27), (7, 6)],
                       'DeepSeek-R1-Distill-Llama-8B': [(10, 27), (28, 8), (12, 21), (12, 2), (20, 9), (27, 28), (14, 10), (14, 1), (10, 9)],
                       'Qwen3-8B': [(17, 5), (32, 11), (16, 19), (20, 9), (25, 8), (17, 5), (20, 19), (27, 8), (26, 28)]}
    prompt_id_list = [1, 2, 3, 4, 5, 6, 7, 8]
    prompt_info_list = {'Llama-2-7b-chat-hf': [(18, 15), (14, 0), (19, 16), (15, 5), (16, 31), (8, 9), (26, 12), (6, 21)],
                        'DeepSeek-R1-Distill-Llama-8B': [(12, 3), (14, 2), (8, 20), (16, 4), (14, 10), (13, 11), (26, 6), (27, 30)],
                        'Qwen3-8B': [(14, 19), (22, 6), (15, 20), (27, 8), (17, 12), (15, 26), (27, 10), (19, 16)]}
    
    model_avg_sim_dict = {}
    model_sim_error_dict = {}
    for model in model_name:
        avg_sim_list = []
        upper_error = []
        lower_error = []
        for prompt_id, info in zip(prompt_id_list, prompt_info_list[model]):
            sim_matrix = compare_direction(layer=info[0], head=info[1], fixed_item=prompt_id, varying='trait', model_name=model)

            mean_sim_matrix = sim_matrix.mean()
            sim_max = np.max(sim_matrix)
            sim_min = np.min(sim_matrix)

            upper_error.append(sim_max - mean_sim_matrix)
            lower_error.append(mean_sim_matrix - sim_min)
            avg_sim_list.append(mean_sim_matrix)
        model_avg_sim_dict[model] = avg_sim_list
        model_sim_error_dict[model] = [lower_error, upper_error]

    patterns = ['//', '+', '\\'] 
    colors = ['#b7b5a0', '#44757a', '#452a3d']
    # x_labels = ['Holistic', 'Content', 'Org', 'WC', 'SF', 'Conv', 'PA', 'Lang', 'Nar']
    x_labels = [f'P{i}' for i in prompt_id_list]

    x = np.arange(len(x_labels))  # 9个trait
    width = 0.25  # 柱状图宽度
    fig, ax = plt.subplots(figsize=(3.3125, 1.65625))

    for i, model in enumerate(model_name):
        avg_sim = model_avg_sim_dict[model]
        lower_error, upper_error = model_sim_error_dict[model]
        
        ax.bar(
            x + i * width,               # 横坐标偏移
            avg_sim,                     # 柱子的高度
            width,                       # 柱子宽度
            label=model,                 # 图例
            color=colors[i],             # 柱子颜色
            edgecolor='white',           # 边框颜色
            yerr=[lower_error, upper_error],  # 上下误差
            capsize=1.5,                   # 误差条端帽长度
            hatch=patterns[i],
            error_kw=dict(elinewidth=0.4, ecolor='black')
        )

    # 设置x轴标签和刻度
    ax.set_ylabel('Average Cosine Similarity', fontsize=7)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x + width)
    xticklabels = ax.set_xticklabels(x_labels, fontsize=5.2)
    # for label in xticklabels:
    #     label.set_horizontalalignment('right')
    ax.tick_params(axis='y', labelsize=6)

    # 添加图例
    ax.legend(fontsize=4, loc='lower left', borderaxespad=0.5, bbox_to_anchor=(0.0, 0.0),)
    plt.tight_layout()
    plt.savefig(f'./AES/ASAP/plot/prompt_direction_avg_sim.png', dpi=300)


