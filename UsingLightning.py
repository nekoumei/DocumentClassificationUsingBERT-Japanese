import pytorch_lightning as pl
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import mojimoji
import re
import collections
import torchtext
from torchtext.data import Field, Dataset, Example
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from logzero import logger
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm
import os
import random

class BERTClaffier(pl.LightningModule):
    def __init__(self, train_df, val_df, net_dir=None, max_length=512, batch_size=32, num_labels=2, num_epochs=100, random_seed=None):
        super(BERTClaffier, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        if random_seed is not None:
            self.seed_everything(random_seed)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
        if net_dir is None:
            self.net = BertForSequenceClassification.from_pretrained('bert-base-japanese-whole-word-masking', num_labels=num_labels)
        else:
            self.net = BertForSequenceClassification.from_pretrained(net_dir)
        self.TEXT = torchtext.data.Field(
            sequential=True,
            tokenize=self.tokenizer_with_preprocessing,
            use_vocab=True,
            lower=False,
            include_lengths=True,
            batch_first=True,
            fix_length=max_length,
            init_token='[CLS]',
            eos_token='[SEP]',
            pad_token='[PAD]',
            unk_token='[UNK]'
        )
        self.LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info('Set random seeds')

    def tokenizer_with_preprocessing(self, text):
        # 半角、全角の変換
        text = mojimoji.han_to_zen(text)
        # 改行、半角スペース、全角スペースを削除
        text = re.sub('\r', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('　', '', text)
        text = re.sub(' ', '', text)
        # 数字文字の一律「0」化
        text = re.sub(r'[0-9 ０-９]', '0', text)  # 数字
        ret = self.tokenizer.tokenize(text)
        return ret

    def _build_vocab_from_dataset(self, ds, min_freq=1):
        logger.info('[Start]Build vocab for TEXT')
        self.TEXT.build_vocab(ds, min_freq=min_freq)
        self.TEXT.vocab.stoi = self.tokenizer.vocab
        logger.info('[Finished]Build vocab for TEXT')
    
    def forward(self, input_ids, labels):
        loss, logit = self.net(input_ids=input_ids, labels=labels)
        yhat = F.softmax(logit, dim=1)
        return loss, yhat

    def training_step(self, batch, batch_nb):
        input_ids = batch.Text[0]
        labels = batch.Label
        
        loss, yhat = self.forward(input_ids, labels)
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        input_ids = batch.Text[0]
        labels = batch.Label
        
        loss, yhat = self.forward(input_ids, labels)
        
        _, yhat = torch.max(yhat, 1)
        val_acc = accuracy_score(yhat.cpu(), labels.cpu())
        val_acc = torch.tensor(val_acc)
        
        return {'val_loss': loss, 'val_acc': val_acc}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    @pl.data_loader
    def train_dataloader(self):
        if not hasattr(self.TEXT, 'vocab'):
            train_ds = DataFrameDataset(self.train_df, fields={'Text': self.TEXT, 'Label': self.LABEL})
            self._build_vocab_from_dataset(train_ds)
        return torchtext.data.Iterator(train_ds, batch_size=self.batch_size, train=True)

    @pl.data_loader
    def val_dataloader(self):
        val_ds = DataFrameDataset(self.val_df, fields={'Text': self.TEXT, 'Label': self.LABEL})
        return torchtext.data.Iterator(val_ds, batch_size=self.batch_size, train=False, sort=False)

class DataFrameDataset(Dataset):
    """
    pandas DataFrameからtorchtextのdatasetつくるやつ
    https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext
    """
    def __init__(self, examples, fields, filter_pred=None):
        """
         Create a dataset from a pandas dataframe of examples and Fields
         Arguments:
             examples pd.DataFrame: DataFrame of examples
             fields {str: Field}: The Fields to use in this tuple. The
                 string is a field name, and the Field is the associated field.
             filter_pred (callable or None): use only exanples for which
                 filter_pred(example) is true, or use all examples if None.
                 Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""
    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex
