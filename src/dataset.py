from transformers import AutoTokenizer


class SiameseDataset(object):
    """
    reference: https://towardsdatascience.com/interpreting-semantic-text-similarity-from-transformer-models-ba1b08e6566c
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('inputs/bert-base-chinese')
    
    def __getitem__(self, inputs):
        for input in inputs:
            query, candidiate = *input[:2]
            