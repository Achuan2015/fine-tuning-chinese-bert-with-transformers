from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import torch.nn as nn

def train(dataloader, model, device, num_epoch=3):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epoch * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    criterion = nn.CosineEmbeddingLoss(margin=0, reduction='mean')
    for _ in range(num_epoch):
        for batch in dataloader:
            optimizer.zero_grad()
            encoded_sent1 = batch['sent1']
            encoded_sent2 = batch['sent2']
            y = batch['label'].to(device)
            output_sent1, output_sent2 = model(encoded_sent1, encoded_sent2)
            # 因为 labels的取值是[0, 1], 而CosineEmbeddingLoss中要求label取值范围在[-1, 1]之间
            # y = 2 * label - 1
            loss = criterion(output_sent1, output_sent2, (2 * y - 1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)

    model.eval()

def test(dataloader, model, device):
    pass