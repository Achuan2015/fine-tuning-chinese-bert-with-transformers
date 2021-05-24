"""
reference: https://huggingface.co/transformers/training.html#trainer
"""



from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_metric

def train(dataloader, model, device, num_epoch=3):
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
            encoded_sent1 = {k: v.to(device) for k, v in batch['sent1'].items()}
            encoded_sent2 = {k: v.to(device) for k, v in batch['sent2'].items()}
            y = batch['label'].to(device)
            output_sent1, output_sent2 = model(encoded_sent1, encoded_sent2)
            # 因为 labels的取值是[0, 1], 而CosineEmbeddingLoss中要求label取值范围在[-1, 1]之间
            # y = 2 * label - 1
            loss = criterion(output_sent1, output_sent2, (2 * y - 1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)

def test(dataloader, model, device):
    THRESHOLD = 0.9
    metric = load_metric('accuracy')
    model.eval()
    for batch in dataloader:
        encoded_eval_sent1 = {k: v.to(device) for k, v in batch['sent1'].items()}
        encoded_eval_sent2 = {k: v.to(device) for k, v in batch['sent2'].items()}
        with torch.no_grad():
            sent1_output = model.encode(encoded_eval_sent1)
            sent2_output = model.encode(encoded_eval_sent2)
        
        sim_score = F.cosine_similarity(sent1_output, sent2_output)
        sim_score = sim_score.cpu()
        predictions = sim_score.apply_(lambda x: 1.0 if x >= THRESHOLD else 0)
        metric.add_batch(predictions=predictions, references=batch['label'])
    result = metric.compute()
    print(result)