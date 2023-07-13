from transformers import  BertPreTrainedModel
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class BertForCoSentMultiLabelNetwork(BertPreTrainedModel):

    def __init__(self, model_path, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(model_path)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.multi_label_loss = nn.BCEWithLogitsLoss()

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
    
    def forward(self, encoded_sent1, encoded_sent2=None, label_ids=None, multi_label_ids=None):
        sent1_embedding = self.encode(encoded_sent1)   # batch_size * hidden_size
        if encoded_sent2 is not None:
            sent2_embedding = self.encode(encoded_sent2)   # batch_size * hidden_size
            sent2_norm_embedding = F.normalize(sent2_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        # cos_score = F.cosine_similarity(sent1_embedding, sent2_embedding, dim=1)
        sent1_norm_embedding = F.normalize(sent1_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        # 既没有二分类标签，也没有多分类标签
        if label_ids is None and multi_label_ids is None:
            return (sent1_norm_embedding, sent2_norm_embedding) if encoded_sent2 is not None else sent1_norm_embedding

        if label_ids is not None:
            loss = self.binary_classification_loss(label_ids, sent1_norm_embedding, sent2_norm_embedding)
        if multi_label_ids is not None:
            loss = self.multi_label_loss(sent1_norm_embedding, multi_label_ids)
        return (sent1_norm_embedding, sent2_norm_embedding, loss) if encoded_sent2 is not None else (sent1_norm_embedding, loss)
        
    def binary_classification_loss(self, label_ids, sent1_embed, sent2_embed, λ=20):
        sent_cosine = torch.sum(sent1_embed * sent2_embed, dim=1) * λ # [batch_size]
        sent_cosine_diff = sent_cosine[:, None] - sent_cosine[None, :]  # 实现 si - sj 的粗结果 (未进行条件 si < sj 的筛选)
        labels = label_ids[:, None] < label_ids[None, ]  # 进行条件 si < sj 的筛选, 不满足条件的都是 False
        # Attention:这里对label的类型进行了转码
        labels = labels.long() # False -> 0, True -> 1
        sent_cosine_exp = sent_cosine_diff - (1 - labels) * 1e12  # 满足条件 True的位置 不变， False的位置直接给一个很小的数类似于 - np.inf 的意思
        # loss function 完成形式 log(1 + ∑log(1 + e^λ(si - sj))) 这里 λ 取值为20
        sent_cosine_exp = torch.cat((torch.zeros(1).to(sent_cosine_exp.device), sent_cosine_exp.view(-1)), dim=0)
        loss =  torch.logsumexp(sent_cosine_exp, dim=0)
        return loss
    
    def multi_label_loss(self, sent_embed, multi_label_ids):
        sent_embed_output = self.dropout(sent_embed)
        logits = self.classifier(sent_embed_output)
        loss = self.multi_label_loss(logits, multi_label_ids)
        return loss
        

class BertForCoSentNetwork(BertPreTrainedModel):

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
    
    def forward(self, encoded_sent1, encoded_sent2=None, label_ids=None, λ=20):
        sent1_embedding = self.encode(encoded_sent1)   # batch_size * hidden_size
        if encoded_sent2 is not None:
            sent2_embedding = self.encode(encoded_sent2)   # batch_size * hidden_size
            sent2_norm_embedding = F.normalize(sent2_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        # cos_score = F.cosine_similarity(sent1_embedding, sent2_embedding, dim=1)
        sent1_norm_embedding = F.normalize(sent1_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        # 既没有二分类标签，也没有多分类标签
        if label_ids is None:
            return (sent1_norm_embedding, sent2_norm_embedding) if encoded_sent2 is not None else sent1_norm_embedding

        sent_cosine = torch.sum(sent1_norm_embedding * sent2_norm_embedding, dim=1) * λ # [batch_size]
        sent_cosine_diff = sent_cosine[:, None] - sent_cosine[None, :]  # 实现 si - sj 的粗结果 (未进行条件 si < sj 的筛选)
        labels = label_ids[:, None] < label_ids[None, ]  # 进行条件 si < sj 的筛选, 不满足条件的都是 False
        # Attention:这里对label的类型进行了转码
        labels = labels.long() # False -> 0, True -> 1
        sent_cosine_exp = sent_cosine_diff - (1 - labels) * 1e12  # 满足条件 True的位置 不变， False的位置直接给一个很小的数类似于 - np.inf 的意思
        # loss function 完成形式 log(1 + ∑log(1 + e^λ(si - sj))) 这里 λ 取值为20
        sent_cosine_exp = torch.cat((torch.zeros(1).to(sent_cosine_exp.device), sent_cosine_exp.view(-1)), dim=0)
        loss =  torch.logsumexp(sent_cosine_exp, dim=0)
        return sent1_norm_embedding, sent2_norm_embedding, loss



class BertForCoSentNetworkTC(BertPreTrainedModel):

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
    
    def forward(self, input_ids, token_type_ids, attention_mask, λ=20):
        encoded_sent1 = {'input_ids': input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}
        sent1_embedding = self.encode(encoded_sent1)   # batch_size * hidden_size
        sent1_norm_embedding = F.normalize(sent1_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        return sent1_norm_embedding
