from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    """
    reference:https://huggingface.co/transformers/custom_datasets.html?highlight=datasets
    """

    def __init__(self, encoded_sent1, encoded_sent2, labels):
        self.encoded_sent1 = encoded_sent1
        self.encoded_sent2 = encoded_sent2
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            'sent1': {key:val[idx] for key, val in self.encoded_sent1.items()},
            'sent2': {key:val[idx] for key, val in self.encoded_sent2.items()},
            'label': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)