from data_util import read_data
from data_util import encode_data
from sklearn.model_selection import train_test_split


def run():
    data_path = "data/sample_50_1.csv"
    sent1, sent2, labels = read_data(data_path)
    encoded_sent1, encoded_sent2, labels = encode_data(sent1, sent2, labels)


if __name__ == "__main__":
    run()