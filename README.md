# Fine-tuning-chinese-bert-with-transformers
Fine-tuning Chinese BERT model based with transformers Trainer api

## Transformer-Trainer 接口
目标是实现 sentence-Transformer 论文中的模型（BertForSiameseNetwork)结果，并进行训练。

参考Transformer的[Trainer](https://huggingface.co/transformers/main_classes/trainer.html)的基本用法[Training and fine-tuning](https://huggingface.co/transformers/training.html#trainer) 中naive PyTorch的基本流程。构造 model, dataset 等对象即可。

## 构造model, dataset的流程
目标：实现基于BERT的SiameseNetwork的Model以及Dataset。
* 读取所有样构造成 InputExample，并保存为List 对象
* 实现Dataset的类：SiameseDataset，输入：examples, tokenizer，实现方法：__getitem__：实现分词并返回token_id 等序列。
* 实现collate_fn(整理样本的函数）: 将样本的List对象初始化为DataLoader（回调），对于每一个batch的数据进行处理（如：query/candidates 长度对齐，并构造自定义Model的输入：）
* 实现自定义Model: BertForSiameseNetWork, 需要继承类 BertPreTrainedModel, 类BertPreTrainedModel，使用BertModel 作为encoder；实现forward的函数：自定义输入特征函数，使用encoder 获得 query/candidates 的句向量，计算余弦相似度作为loss 返回。
* 实现 compute_metric, 进行指标计算
* 初始化Trainer 实例，输入BertForSiameseNetwork, SiameseDataset, collate_fn以及compute_metrics。
