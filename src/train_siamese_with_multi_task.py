import torch
from data_util import read_data, read_label_data, read_atec_data
from data_util import encode_data, encode_label_data 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import BertForCoSentMultiLabelNetwork
from dataset import SiameseDataset, SiameseLabelDataset
from transformers import BertConfig
from engine import train_siamese_with_cosent
from engine import test_siamese_with_cosent
from engine import train_siamese_with_multi_label
from transformers import BertTokenizer


def run():
    """
    reference: https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """
    torch.cuda.set_device(1)
    output_dir = "outputs/TinyBert-15-1-cosent-bot-v5"
    data_path = "/data/projects/negative-sample-with-KMeans/output_data/bot_train_20220620_v1_15_1.csv"
    multi_label_data_path = "/data/projects/negative-sample-with-KMeans/output_data/data_label_corpus_20220620_v2.csv"
    model_path = "/data/projects/TinyBERT_4L_zh"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print("当前使用GPU :", torch.cuda.current_device())

    text, multi_label = read_label_data(multi_label_data_path)
    sent1, sent2, labels = read_data(data_path)

    train_sent1, eval_sent1, train_sent2, eval_sent2, train_label, eval_label = train_test_split(sent1, sent2, labels, test_size=0.1, random_state=42)
    train_text, eval_text, train_multi_label, eval_multi_label = train_test_split(text, multi_label, test_size=0.1, random_state=42)
    
    print('开始预处理二分类数据')
    encoded_train_sent1, encoded_train_sent2, train_label= encode_data(train_sent1, train_sent2, train_label, tokenizer)
    encoded_eval_sent1, encoded_eval_sent2, eval_label= encode_data(eval_sent1, eval_sent2, eval_label, tokenizer)
    print('结束二分类数据处理')
    
    print('开始预处理多标签数据')
    encoded_train_text, train_multi_label = encode_label_data(train_text, train_multi_label, tokenizer)
    encoded_eval_text, eval_multi_label = encode_label_data(eval_text, eval_multi_label, tokenizer)
    print('结束处理多标签数据')

    train_dataset = SiameseDataset(encoded_train_sent1, encoded_train_sent2, train_label)
    eval_dataset = SiameseDataset(encoded_eval_sent1, encoded_eval_sent2, eval_label)
    
    train_label_dataset =  SiameseLabelDataset(encoded_train_text, train_multi_label)
    eval_label_dataset = SiameseLabelDataset(encoded_eval_text, eval_multi_label)

    print('开始构造 dataloader') 
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True) 

    train_label_dataloader = DataLoader(train_label_dataset, batch_size=128, shuffle=True)
    eval_label_dataloader = DataLoader(eval_label_dataset, batch_size=32, shuffle=True) 

    print('完成构造 dataloader')
    num_labels = 84
    config = BertConfig.from_pretrained(model_path, num_labels=num_labels)
    model = BertForCoSentMultiLabelNetwork(model_path, config)
    model = model.to(device)
    
    print('开始训练 二分类 部分')
    train_siamese_with_cosent(train_dataloader, model, device)
    print('结束 二分类部分 训练')
    
    print('开始训练 多标签 部分')
    train_siamese_with_multi_label(train_label_dataloader, model, device)
    print('结束 多标签 部分训练')

    print('开始验证')
    test_siamese_with_cosent(eval_dataloader, model, device)
    print('结束验证') 
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def eval():
    model_path = "outputs/TinyBert-15-1-cosent-bot-v5"
    data_path = "/data/data-lakes/atec/atec_dataset_test.csv"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)

    sent1, sent2, labels = read_atec_data(data_path)
    encoded_sent1, encoded_sent2, encoded_label= encode_data(sent1, sent2, labels, tokenizer)
    dataset = SiameseDataset(encoded_sent1, encoded_sent2, encoded_label)
    dataloader = DataLoader(dataset, batch_size=192, shuffle=True)

    config = BertConfig.from_pretrained(model_path)
    model = BertForCoSentMultiLabelNetwork(model_path, config)
    model = model.to(device)

    test_siamese_with_cosent(dataloader, model, device)


if __name__ == "__main__":
    # run()
    eval()
