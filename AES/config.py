import configargparse

def load_config():
    parser = configargparse.ArgumentParser(description = "main")

    parser.add_argument('--train_path', default='./AES/ASAP/data.tsv', type=str, help='Train data file path')
    parser.add_argument('--prompt_path', default='./AES/Json/Prompt.json', type=str, help='Prompt text file path')
    parser.add_argument('--index_path', default='./AES/Json/Index.json', type=str, help='Index file path')

    parser.add_argument('--bert_path', default='./model/bert-base-uncased', type=str, help='Bert model path')
    parser.add_argument('--model_path', default='./model/', type=str, help='Model path')
    parser.add_argument('--model_name', default='Llama-2-7b-chat-hf', type=str, help='LLM model name')

    args = parser.parse_args()

    return args