import torch
from transformers import BertTokenizer
from transformers import BertConfig

from model import BertForCoSentNetwork, BertForCoSentNetworkTC


def convert2torchscript(): 
    MAX_LENGTH = 256
    model_path = "outputs/TinyBert-15-1-cosent-bot-v4"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path, torchscript=True)
    model = BertForCoSentNetwork(model_path, config)
    model.to(device)
    
    query1 = "您好，您看这样处理可以吗？"
    query2 = "主任，您真是心怀患者"
    encoded_query1 = tokenizer(query1, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query1 = {k:v.to(device) for k, v in encoded_query1.items()}
    encoded_query2 = tokenizer(query2, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query2 = {k:v.to(device) for k, v in encoded_query2.items()}
    
    model.eval()
    traced_model = torch.jit.trace(model, [encoded_query1])
    torch.jit.save(traced_model, "traced_tinybert_bot_v4-1.pt")

def convert2torchscript_v2():
    MAX_LENGTH = 256
    model_path = "outputs/TinyBert-15-1-cosent-bot-v4"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path, torchscript=True)
    model = BertForCoSentNetworkTC(model_path, config)
    model.to(device)
    
    query1 = "您好，您看这样处理可以吗？"
    encoded_query1 = tokenizer(query1, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query1 = {k:v.to(device) for k, v in encoded_query1.items()}
    
    model.eval()
    query_input = list(encoded_query1.values())
    traced_model = torch.jit.trace(model, query_input)
    torch.jit.save(traced_model, "traced_tinybert_bot_v4-2.pt")

def test():
    MAX_LENGTH = 256
    model_path = "outputs/TinyBert-15-1-cosent-bot-v4"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path, torchscript=True)
    model = BertForCoSentNetwork(model_path, config)
    model = model.to(device)
    
    query1 = "您好，您看这样处理可以吗？"
    query2 = "主任，您真是心怀患者"
    encoded_query1 = tokenizer(query1, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query1 = {k:v.to(device) for k, v in encoded_query1.items()}
    encoded_query2 = tokenizer(query2, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query2 = {k:v.to(device) for k, v in encoded_query2.items()}
    
    model.eval()
    response = model(encoded_query1, encoded_query2)
    response = [i.detach().cpu().numpy().tolist()[0] for i in response]
    print(response)

def test_script_model():
    MAX_LENGTH = 256
    model_path = "outputs/TinyBert-15-1-cosent-bot-v4"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = torch.jit.load("traced_torchscript_tinybert_bot_v4.pt")
    model = model.to(device)
    
    query1 = "您好，您看这样处理可以吗？"
    query2 = "主任，您真是心怀患者"
    encoded_query1 = tokenizer(query1, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query1 = {k:v.to(device) for k, v in encoded_query1.items()}
    encoded_query2 = tokenizer(query2, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded_query2 = {k:v.to(device) for k, v in encoded_query2.items()}
    
    model.eval()
    response = model(encoded_query1, encoded_query2)
    response = [i.detach().cpu().numpy().tolist()[0] for i in response]
    print(response)


if __name__ == '__main__':
    convert2torchscript_v2()
    # # test()
    # test_script_model()
