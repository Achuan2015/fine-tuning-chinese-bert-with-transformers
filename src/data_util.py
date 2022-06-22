import pandas as pd
import numpy as np
import torch


def read_label_data(path):
    dfs = pd.read_csv(path, sep='\t')
    dfs['label'] = dfs['one_hot_label'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' ').tolist())
    text = dfs['text'].tolist()
    one_hot_labels = dfs['label'].tolist()
    return text, one_hot_labels

def read_data(path):
    dfs = pd.read_csv(path, delimiter='\t')
    querys = dfs['query'].tolist()
    candidates = dfs['candidate'].tolist()
    labels = pd.to_numeric(dfs["label"]).tolist()
    return querys, candidates, labels


def read_atec_data(path):
    dfs = pd.read_csv(path, sep='\t', header=None, names=['id', 'query', 'candidate', 'label'])
    querys = dfs['query'].tolist()
    candidates = dfs['candidate'].tolist()
    labels = pd.to_numeric(dfs["label"]).tolist()
    return querys, candidates, labels
    
def encode_label_data(text, label, tokenizer):
    encoded_text = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    labels = torch.FloatTensor(label)
    return encoded_text, labels

def encode_data(querys, candidates, labels, tokenizer):
    encoded_query = tokenizer(querys, padding=True, truncation=True, max_length=256, return_tensors="pt")
    encoded_candidate = tokenizer(candidates, padding=True, truncation=True, max_length=256, return_tensors="pt")
    labels = torch.tensor(labels)
    return encoded_query, encoded_candidate, labels


def encode_pairs_sentence(sent1, sent2, labels, tokenizer):
    encoded_data = tokenizer(sent1, sent2, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.float)
    return encoded_data, labels
