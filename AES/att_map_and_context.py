import re
import utils
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from baukit import Trace
from qwk_cal import qwk_calculation
from probe import MLP_Regressor
from sklearn.linear_model import Ridge
from config import load_config
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib.colors import LinearSegmentedColormap



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


def get_spec_essay(args, prompt_id, trait, essay_id, prompt_type='all'):
    prompt = utils.load_prompt(args.prompt_path, prompt_id=prompt_id)
    essays, _ = utils.load_essay_data(args.train_path)
    prompt_essays = essays[prompt_id - 1]
    essay = prompt_essays[essay_id]
    content = create_content(prompt, essay, trait, prompt_type)
    return content


prompt_dict = {
    'Content': 'Please evaluate the above essay from the perspective of the quantity of content and ideas presented in the article.',
    'Organization': 'Please evaluate the above essay from the perspective of the organization of ideas and structure of the article.',
    'Word Choice': 'Please evaluate the above essay from the perspective of the choice of words and sentences.',
    'Sentence Fluency': 'Please evaluate the above essay from the perspective of the fluency of sentences.',
    'Conventions': 'Please evaluate the above essay from the perspective of the use of conventions and conventions of style.',
    'Prompt Adherence': 'Please evaluate the above essay from the perspective of the adherence to the prompt.',
    'Language': 'Please evaluate the above essay from the perspective of the language used.',
    'Narrativity': 'Please evaluate the above essay from the perspective of the narrativity of the writing.',
    'Holistic': 'Please evaluate the above essay.'
}
def create_content(prompt, essay, trait, type):
    assert type in ['all', 'wo_p', 'wo_i', 'only_e']
    if type == 'all':
        return 'Prompt: ' + prompt + '\n\n' + 'Essay: ' + essay + '\n\n' + prompt_dict[trait]
    elif type == 'wo_p':
        return 'Essay: ' + essay + '\n\n' + prompt_dict[trait]
    elif type == 'wo_i':
        return 'Prompt: ' + prompt + '\n\n' + 'Essay: ' + essay
    elif type == 'only_e':
        return 'Essay: ' + essay

def tokenize_prompt(prompt, tokenizer):
    tokenized_prompt = tokenizer.encode(prompt, return_tensors='pt')
    return tokenized_prompt

def get_head_wise_qkv_attention(content, layer, head, model, device, tokenizer, draw_part=False, draw_full=False):

    tokenized_prompt = tokenize_prompt(content, tokenizer)
    token_to_word, words = get_token_to_word_map(content, tokenizer)
    output = model(tokenized_prompt.to(device), output_attentions=True)
    qkv_attention = output.attentions[layer][0, head, :, :].detach().cpu().numpy()
    word_attention = token_attention_to_word_attention(qkv_attention, token_to_word, len(words))
    print(f'Attention shape: {word_attention.shape} Token size: {len(words)}')

    if draw_part:
        split_points = [word_attention.shape[0] // 3, 2 * (word_attention.shape[0] // 3)]
        rows = np.split(word_attention, split_points, axis=0)
        blocks_2d = [np.split(row, split_points, axis=1) for row in rows]
        blocks = [block for row in blocks_2d for block in row]

        word_segments = np.split(np.array(words), split_points)
        word_blocks = [seg for seg in word_segments]


        for ii, part in enumerate(blocks):
            plt.figure(figsize=(30, 30))  # 控制图像大小
            plt.imshow(part, cmap='Greys', vmax=word_attention.max(), vmin=word_attention.min())
            plt.colorbar(label="Attention Weight")
            plt.title("Attention Map")

            # 设置坐标轴标签
            x_labels = word_blocks[ii % 3]
            y_labels = word_blocks[ii // 3]
            
            plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=90)
            plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

            plt.xlabel("Key Token Index")
            plt.ylabel("Query Token Index")
            plt.tight_layout()
            plt.savefig(f"attention_map_{ii+1}.png")
            plt.close()

    if draw_full:
        plt.figure(figsize=(20, 20))
        plt.imshow(word_attention, cmap='Greys', vmax=0.2 * word_attention.max(), vmin=word_attention.min())
        plt.colorbar(label="Attention Weight")
        plt.title("Full Attention Map")

        # 设置完整 token 标签
        plt.xticks(ticks=np.arange(len(words)), labels=words, rotation=90)
        plt.yticks(ticks=np.arange(len(words)), labels=words)

        plt.xlabel("Key Token Index")
        plt.ylabel("Query Token Index")
        plt.tight_layout()
        plt.savefig("attention_map_full.png")
        plt.close()

    return token_to_word, words

def get_token_to_word_map(text, tokenizer):
    words = re.findall(r'\S+|\n', text)
    token_to_word = []
    for word_idx, word in enumerate(words):
        if word == '\n':
            words[word_idx] = '<0X0A>'
            token_to_word.extend([word_idx + 1])
            continue
        elif word == 'Essay:':
            token_to_word.extend([word_idx + 1] * 4)
            continue
        tokenized_word = tokenizer.tokenize(word)
        token_to_word.extend([word_idx + 1] * len(tokenized_word))
    return [0] + token_to_word, ['<s>'] + words

def token_attention_to_word_attention(token_attention, token_to_word, num_words):
    word_attention = np.zeros((num_words, num_words))
    word_counts = np.zeros((num_words, num_words))

    for i in range(len(token_to_word)):
        for j in range(len(token_to_word)):
            wi = token_to_word[i]
            wj = token_to_word[j]
            word_attention[wi][wj] += token_attention[i][j]
            word_counts[wi][wj] += 1

    word_attention = np.divide(word_attention, word_counts, out=np.zeros_like(word_attention), where=word_counts!=0)
    return word_attention



def split_data_cross_prompt(data, label, prompt_id):
    train_data_list = [d for i, d in enumerate(data) if isinstance(d, torch.Tensor) and i != (prompt_id - 1)]
    train_label_list = [l for i, l in enumerate(label) if isinstance(l, torch.Tensor) and i != (prompt_id - 1)]
    train_data = torch.cat(train_data_list, dim=0)
    train_label = torch.cat(train_label_list, dim=0)
    test_data = data[prompt_id - 1]
    test_label = label[prompt_id - 1]
    return train_data, test_data, train_label, test_label

def spec_head_regression(prompt_id, trait, layer, head, model_name, probe_type='Linear'):
    index = utils.load_json_file('./AES/Json/Index.json')
    head_wise_activations = [torch.load(f'./AES/ASAP/activations/{model_name}/{model_name}_Prompt_{ii}_{trait}_all.pt') 
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
        probe = MLP_Regressor(hidden_layer_sizes=(128, ), activation='relu', max_iter=500, random_state=42)
    probe.fit(train_data[:, layer, head, :], train_label)
    y_pred = probe.predict(test_data[:, layer, head, :])
    y_pred = np.clip(y_pred, 0, 1)      
    y_pred = np.rint(diff * y_pred + bias).astype(int) 
    y_pred = torch.tensor(y_pred, dtype=torch.int32)
    score = qwk_calculation(test_label, y_pred)
    print(f'Probe predict score: {score:.4f}')
    return probe

def compute_word_avg_score(token_to_word, score, word_count):
    word_score_sum = np.zeros(word_count)
    word_token_count = np.zeros(word_count)

    for t_idx, w_idx in enumerate(token_to_word):
        word_score_sum[w_idx] += score[t_idx]
        word_token_count[w_idx] += 1

    word_avg_score = word_score_sum / np.maximum(word_token_count, 1)

    return word_avg_score



def get_spec_essay_activation(content, layer, head, model, device, tokenizer):

    tokenized_prompt = tokenize_prompt(content, tokenizer)
    head_wise_activation = get_activation(model, tokenized_prompt, device, layer, head)
    print(f"activation's shape {head_wise_activation.shape}")
    return head_wise_activation

def get_llama_activations_bau(model, prompt, device, target): 
    model.eval()

    with torch.no_grad():
        prompt = prompt.to(device)
        with Trace(model, target, retain_input=True) as ret:
            _ = model(prompt, output_hidden_states = True)
        head_wise_hidden_states = ret.input.squeeze().detach()
    return head_wise_hidden_states

def get_activation(model, prompt, device, layer, head):
    HEAD = f"model.layers.{layer}.self_attn.o_proj"
    head_wise_activations = get_llama_activations_bau(model, prompt, device, HEAD)
    head_wise_activations = head_wise_activations.reshape(head_wise_activations.shape[0], 32, 128)
    head_wise_activation = head_wise_activations[:, head, :]
    return head_wise_activation.cpu().numpy()



def visualize_word_scores(words, scores, cmap='bwr', figsize=(12, 5), font_size=12):
    """
    words: List[str] - 每个词
    scores: List[float] - 每个词的分数，正负代表不同颜色方向
    cmap: str - 颜色映射名
    """

    assert len(words) == len(scores), "词和分数数量不一致"

    norm = colors.Normalize(vmin=min(scores), vmax=max(scores))
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    x = 0.01
    y = 1
    line_height = 0.08
    max_width = 0.95

    for word, score in zip(words, scores):
        if word == '<s>':
            continue
        elif word == '<0X0A>':
            y -= line_height
            x = 0.01
            continue
        elif word == 'Please':
            break

        color = cmap(norm(score)) if score > 0.5 else 'none' 
        ax.text(x, y, word + ' ', fontsize=font_size, bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2'))

        text_width = (len(word) + 1) * 0.01 

        x += text_width
        if x > max_width:
            x = 0.01
            y -= line_height

    fig.savefig('word_scores.png', bbox_inches='tight')


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name, use_fast=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(args.model_path + args.model_name, torch_dtype=torch.float16, attn_implementation="eager").to(device=device).eval()
    return model, device, tokenizer


if __name__ == '__main__':
    args = load_config()

    prompt_id = 4
    trait = 'Content'
    essay_id = 15
    layer = 13
    head = 28
    prompt_type = 'all'
    model_name = 'Llama-2-7b-chat-hf'
    head_info = utils.load_json_file('Llama-2-7b-chat-hf_cross_prompt_Linear_sorted_model_head_info.json')

    index = utils.load_json_file('./AES/Json/Index.json')
    diff = max_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]] - min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]
    bias = min_values[prompt_id - 1][index[f'Prompt_{prompt_id}'][trait]]

    content = get_spec_essay(args, prompt_id=prompt_id, trait=trait, essay_id=essay_id, prompt_type=prompt_type)
    print(content)
    model, device, tokenizer = load_model(args)

    score = []
    for info in head_info[trait][f'Prompt_{prompt_id}'][:8]:
        token_to_word, words = get_head_wise_qkv_attention(content=content, layer=info['layer'], head=info['head'], 
                                                           model=model, device=device, tokenizer=tokenizer, draw_full=False, draw_part=False)
        print(f'Token num: {len(token_to_word)} Words num: {len(words)}')
        head_wise_activation = get_spec_essay_activation(content=content, layer=info['layer'], head=info['head'],
                                                         model=model, device=device, tokenizer=tokenizer)
        probe = spec_head_regression(prompt_id=prompt_id, trait=trait, layer=info['layer'], head=info['head'], model_name=model_name)

        prediction = probe.predict(head_wise_activation)
        prediction = np.clip(prediction, 0, 1)      

        word_avg_score = compute_word_avg_score(token_to_word, prediction, len(words))
        score.append(word_avg_score)
        # visualize_word_scores(words, word_avg_score)

    score = np.vstack(score)
    colors = [(1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("white_red", colors, N=256)
    height = max(2, score.shape[0] * 0.1)
    plt.figure(figsize=(10, height))
    ax = sns.heatmap(score, cmap=cmap, vmin=0.5, vmax=1, linewidths=0.3)
    
    # for y in range(1, score.shape[0]):
    #     ax.axhline(y=y, color='black', linewidth=0.5)

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.5)

    x_ticks = np.arange(0, len(score[0]), 10)
    plt.xticks(ticks=x_ticks + 0.5, labels=[str(i+1) for i in x_ticks], fontsize=12, rotation=0)
    y_ticks = np.arange(0, 8, 1)
    plt.yticks(ticks=y_ticks + 0.5, labels=[str(i+1) for i in y_ticks], fontsize=12, rotation=0)
    plt.xlabel("Words", fontsize=15)
    plt.ylabel("Heads", fontsize=15)
    # plt.axis('off')
    plt.savefig(f'head_heatmap.png', dpi=300, bbox_inches='tight')




