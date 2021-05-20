from transformers import AdamW
import torch

def train(dataloader, model, epoch=3):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optim = AdamW(model.parameers(), lr=5e-5)
    for _ in range(epoch):
        for batch in dataloader:
            optim.zero_grad()
            input_ids = batch['sent1']
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()

def test(*args):
    pass