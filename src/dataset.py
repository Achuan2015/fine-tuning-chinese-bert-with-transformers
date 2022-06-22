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


class SiameseLabelDataset(Dataset):

    def __init__(self, encoded_text, labels):
        self.encoded_text = encoded_text
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            'text': {key:val[idx] for key, val in self.encoded_text.items()},
            'label': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)


class CrossEncodeDataset(Dataset):
    
    def __init__(self, encoded_data, labels):
        self.encoded_data = encoded_data
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoded_data.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)