from transformers import  BertPreTrainedModel
from transformers import BertModel
import torch
import torch.nn as nn
import torch.functional as F


class BertForSiameseNetwork(BertPreTrainedModel):
    """
    reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
    """

    def __init__(self, model_path, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(model_path)

    def encode(self, encoded_input):
        model_output = self.bert(**encoded_input)
        input_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return input_embedding
    
    def mean_pooling(self, model_output, attention_mask):
        token_embedding = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        # sum columns
        sum_embddings = torch.sum(token_embedding * input_mask_expanded, 1) 
        # sum_mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embddings / sum_mask
    
    def forward(self, encoded_sent1, encoded_sent2):
        sent1_embedding = self.encode(encoded_sent1)
        sent2_embedding = self.encode(encoded_sent2)
        # cos_score = F.cosine_similarity(sent1_embedding, sent2_embedding, dim=1)
        return sent1_embedding, sent2_embedding