import torch
from data_util import read_data, read_atec_data
from data_util import encode_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import BertForCoSentNetwork
from dataset import SiameseDataset
from transformers import BertConfig
from engine import train_siamese_with_cosent
from engine import test_siamese_with_cosent
from transformers import BertTokenizer


def run():
    """
    reference: https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """
    output_dir = "outputs/TinyBert-58-1-cosent-huaxia-v2"
    #output_dir = "outputs/TinyBert-58-1-cosent-huaxia"
    #output_dir = "outputs/chinese_wwm_pytorch-58-1-cosent"
    data_path = "data/huaxia_faq_58_1.csv"
    # model_path = "inputs/chinese_wwm_pytorch"
    model_path = "/data/projects/TinyBERT_4L_zh"
    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print("当前使用GPU :", torch.cuda.current_device())
    sent1, sent2, labels = read_data(data_path)
    train_sent1, eval_sent1, train_sent2, eval_sent2, train_label, eval_label =  train_test_split(sent1, sent2, labels, test_size=0.1, random_state=42)
    print('开始预处理数据')
    encoded_train_sent1, encoded_train_sent2, train_label= encode_data(train_sent1, train_sent2, train_label, tokenizer)
    encoded_eval_sent1, encoded_eval_sent2, eval_label= encode_data(eval_sent1, eval_sent2, eval_label, tokenizer)
    print('结束预处理数据')    
    train_dataset = SiameseDataset(encoded_train_sent1, encoded_train_sent2, train_label)
    eval_dataset = SiameseDataset(encoded_eval_sent1, encoded_eval_sent2, eval_label)
    print('开始构造 dataloader') 
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True) 
    print('完成构造 dataloader')
    config = BertConfig.from_pretrained(model_path)
    model = BertForCoSentNetwork(model_path, config)
    model = model.to(device)
    print('开始训练')
    train_siamese_with_cosent(train_dataloader, model, device)
    print('结束训练')
    print('开始验证')
    test_siamese_with_cosent(eval_dataloader, model, device)
    print('结束验证') 
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def eval():
    model_path = "outputs/TinyBert-50-1-cosent"
    #model_path = "outputs/TinyBert-58-1-cosent-huaxia"
    #model_path = "outputs/chinese_wwm_pytorch-50-1-cosent"
    model_path = "outputs/chinese_wwm_pytorch-50-1-cosent"
    data_path = "/data/data-lakes/atec/atec_dataset_test.csv"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)

    sent1, sent2, labels = read_atec_data(data_path)
    encoded_sent1, encoded_sent2, encoded_label= encode_data(sent1, sent2, labels, tokenizer)
    dataset = SiameseDataset(encoded_sent1, encoded_sent2, encoded_label)
    dataloader = DataLoader(dataset, batch_size=192, shuffle=True)

    config = BertConfig.from_pretrained(model_path)
    model = BertForCoSentNetwork(model_path, config)
    model = model.to(device)

    test_siamese_with_cosent(dataloader, model, device)


if __name__ == "__main__":
    run()
    #eval()
