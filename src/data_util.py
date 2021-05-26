import pandas as pd
import torch


def read_data(path):
    dfs = pd.read_csv(path, delimiter='\t')
    querys = dfs['query'].tolist()
    candidates = dfs['candidate'].tolist()
    labels = pd.to_numeric(dfs["label"]).tolist()
    return querys, candidates, labels

def encode_data(querys, candidates, labels, tokenizer):
    encoded_query = tokenizer(querys, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoded_candidate = tokenizer(candidates, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels = torch.tensor(labels)
    return encoded_query, encoded_candidate, labels

def encode_pairs_sentence(sent1, sent2, labels, tokenizer):
    encoded_data = tokenizer(sent1, sent2, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.float)
    return encoded_data, labels