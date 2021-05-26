import torch
from data_util import read_data
from data_util import encode_pairs_sentence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from dataset import CrossEncodeDataset
from engine import train_cross_encoder
from engine import test_cross_encoder
from transformers import BertTokenizer


def run():
    """
    reference: (1) https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews
               (2) https://huggingface.co/transformers/preprocessing.html
               (3) https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    """
    output_dir = "outputs/TinyBert-classification-50-1"
    data_path = "data/sample_50_1.csv"
    # model_path = "inputs/chinese_wwm_pytorch"
    model_path = "/data/projects/TinyBERT_4L_zh"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)


    sent1, sent2, labels = read_data(data_path)
    # Preprocessing pairs of sentences
    train_sent1, eval_sent1, train_sent2, eval_sent2, train_label, eval_label =  train_test_split(sent1, sent2, labels, test_size=0.1, random_state=42)
    
    encoded_train_data, encoded_train_label =  encode_pairs_sentence(train_sent1, train_sent2, train_label, tokenizer)
    encoded_eval_data, encoded_eval_label =  encode_pairs_sentence(eval_sent1, eval_sent2, eval_label, tokenizer)
    
    train_dataset = CrossEncodeDataset(encoded_train_data, encoded_train_label)
    eval_dataset = CrossEncodeDataset(encoded_eval_data, encoded_eval_label)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)
    
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          num_labels=1)               # regression output
    model = model.to(device)

    train_cross_encoder(train_dataloader, model, device)
    test_cross_encoder(eval_dataloader, model, device)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    run()
