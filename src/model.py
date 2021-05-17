from transformers import  BertPreTrainedModel
from transformers import BertModel
import torch
import torch.nn as nn


class BertForSiameseNetwork(BertPreTrainedModel):
    """
    reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
    """

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cos = nn.CosineSimilarity()
    
    def menn_pooling(self, model_output, attention_mask):
        token_embedding = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        # sum columns
        sum_embddings = torch.sum(token_embedding * input_mask_expanded, 1) 
        # sum_mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embddings / sum_mask
    
    def forward(self, query_ids=None, 
                query_attention_mask=None, 
                candidate_ids=None, 
                candidate_attention_mask=None):
        query_outputs = self.bert(query_ids, query_attention_mask)
        candidate_output = self.bert(candidate_ids, candidate_attention_mask)
        query_pooled_output = self.mean_pooling(query_outputs)
        candidate_pooled_output = self.mean_pooling(candidate_output)
        loss = self.cos(query_pooled_output, candidate_pooled_output)
        return loss