import torch
from data_util import read_data
from data_util import encode_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import BertForSiameseNetwork
from dataset import SiameseDataset
from transformers import BertConfig
from engine import train
from engine import test
from transformers import BertTokenizer


def run():
    """
    reference: https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """
    output_dir = "outputs/TinyBert-50-1"
    data_path = "data/sample_50_1.csv"
    # model_path = "inputs/chinese_wwm_pytorch"
    model_path = "/data/projects/TinyBERT_4L_zh"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)

    sent1, sent2, labels = read_data(data_path)
    train_sent1, eval_sent1, train_sent2, eval_sent2, train_label, eval_label =  train_test_split(sent1, sent2, labels, test_size=0.1, random_state=42)
    encoded_train_sent1, encoded_train_sent2, train_label= encode_data(train_sent1, train_sent2, train_label, tokenizer)
    encoded_eval_sent1, encoded_eval_sent2, eval_label= encode_data(eval_sent1, eval_sent2, eval_label, tokenizer)
    
    train_dataset = SiameseDataset(encoded_train_sent1, encoded_train_sent2, train_label)
    eval_dataset = SiameseDataset(encoded_eval_sent1, encoded_eval_sent2, eval_label)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)
    
    config = BertConfig.from_pretrained(model_path)
    model = BertForSiameseNetwork(model_path, config)
    model = model.to(device)

    train(train_dataloader, model, device)
    test(eval_dataloader, model, device)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    run()