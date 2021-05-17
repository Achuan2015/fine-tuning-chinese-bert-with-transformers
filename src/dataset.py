from transformers import AutoTokenizer


class SiameseDataset(object):
    """
    reference: https://towardsdatascience.com/interpreting-semantic-text-similarity-from-transformer-models-ba1b08e6566c
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('inputs/bert-base-chinese')
    
    def __getitem__(self, inputs):
        input_examples = []
        input_labels = []
        for input in inputs:
            query, candidate, label = input
            encoded_query = self.tokenizer(query, padding=True, truncation=True, max_length=128, return_tensor="pt")
            encoded_candidate = self.tokenizer(candidate, padding=True, truncation=True, max_length=128, return_tensor="pt")
            input_examples.append((encoded_query, encoded_candidate))
            input_labels.append(label)
        return input_examples, input_labels