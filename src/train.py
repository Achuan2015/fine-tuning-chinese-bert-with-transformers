from data_util import read_data
from data_util import encode_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from siamese_dataset import SiameseDataset


def run():
    """
    reference: https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """
    data_path = "data/sample_50_1.csv"
    sent1, sent2, labels = read_data(data_path)
    train_sent1, eval_sent1, train_sent2, eval_sent2, train_label, eval_label =  train_test_split(sent1, sent2, labels, test_size=0.1, random_state=42)
    encoded_train_sent1, encoded_train_sent2, train_label = encode_data(train_sent1, train_sent2, train_label)
    encoded_eval_sent1, encoded_eval_sent2, eval_label = encode_data(eval_sent1, eval_sent2, eval_label)
    train_dataset = SiameseDataset(encoded_train_sent1, encoded_train_sent2, train_label)
    eval_dataset = SiameseDataset(encoded_eval_sent1, encoded_eval_sent2, eval_label)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    run()