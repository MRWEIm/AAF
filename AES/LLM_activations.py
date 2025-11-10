import os
import numpy as np
from tqdm import tqdm
import utils
import torch
from config import load_config
from transformers import AutoTokenizer, AutoModelForCausalLM 
from baukit import TraceDict

def get_essay_activations(model, prompt, device, target_list): 
    model.eval()

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, target_list, retain_input=True) as ret:
            _ = model(prompt, output_hidden_states = True)
        head_wise_hidden_states = [ret[head].input.squeeze().detach() for head in target_list]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze()
        head_wise_hidden_states = head_wise_hidden_states[:, -1, :]
    return head_wise_hidden_states

def get_batch_activations(model, prompts, device):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]

    all_head_wise_activations = []
    for p in tqdm(prompts):
        head_wise_activations = get_essay_activations(model, p, device, HEADS)
        all_head_wise_activations.append(head_wise_activations.cpu())
    head_wise_activations = torch.tensor(np.array(all_head_wise_activations))
    head_wise_activations = head_wise_activations.reshape(head_wise_activations.shape[0], head_wise_activations.shape[1], head_wise_activations.shape[2] // 128, 128)
    return head_wise_activations

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

def tokenize_prompt(prompts, tokenizer):
    tokenized_prompts = []
    for p in prompts:
        p = tokenizer.encode(p, return_tensors='pt')
        tokenized_prompts.append(p)
    return tokenized_prompts

def save_activations(args, prompt_id, trait, type, model, tokenizer, device):
    essays, _ = utils.load_essay_data(args.train_path)
    prompt_essays = essays[prompt_id - 1]
    prompt = utils.load_prompt(args.prompt_path, prompt_id=prompt_id)

    activations_folder_path = './AES/ASAP/activations'
    model_folder = f'/{args.model_name}'
    file_name = f'{args.model_name}_Prompt_{prompt_id}_{trait}_{type}.pt'

    base_path = os.path.join(activations_folder_path, model_folder)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    head_path = os.path.join(base_path, file_name)
    if os.path.exists(head_path):
        head_wise_activations = torch.load(head_path)
    else:
        contents = []
        for essay in prompt_essays:
            content = create_content(prompt, essay, trait, type)
            contents.append(content)
    
        tokenized_prompts = tokenize_prompt(contents, tokenizer)
        head_wise_activations = get_batch_activations(model, tokenized_prompts, device)
        torch.save(head_wise_activations, head_path)
        print(head_wise_activations.shape)

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name, use_fast=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(args.model_path + args.model_name, torch_dtype=torch.float16).to(device=device).eval()
    return model, tokenizer, device


if __name__ == '__main__':
    args = load_config()
    args.model_name = 'Llama-2-7b-chat-hf'

    model, tokenizer, device = load_model(args)
    # print(model)

    type_list = ['all', 'wo_p', 'wo_i', 'only_e']
    trait_list = ['Holistic', 'Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Prompt Adherence', 'Language', 'Narrativity']
    index = utils.load_json_file('./AES/Json/Index.json')
    for type in type_list:
        for prompt_id in range(1, 9):
            if type in ['wo_i', 'only_e']:
                print(f'Processing Prompt_{prompt_id}')
                save_activations(args, prompt_id=prompt_id, trait='None', type=type, model=model, tokenizer=tokenizer, device=device)
            else:
                for trait in trait_list:
                    if trait not in index[f'Prompt_{prompt_id}']:
                        continue
                    print(f'Processing Prompt_{prompt_id} {trait}')
                    save_activations(args, prompt_id=prompt_id, trait=trait, type=type, model=model, tokenizer=tokenizer, device=device)

