import pandas as pd
import torch
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('src/inputs/chinese_wwm_pytorch')

def read_data(path):
    dfs = pd.read_csv(path, delimiter='\t')
    querys = dfs['query'].tolist()
    candidates = dfs['candidate'].tolist()
    labels = pd.to_numeric(dfs["label"]).tolist()
    return querys, candidates, labels

def encode_data(querys, candidates, labels):
    encoded_query = tokenizer(querys, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoded_candidate = tokenizer(candidates, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels = torch.tensor(labels)
    return encoded_query, encoded_candidate, labels