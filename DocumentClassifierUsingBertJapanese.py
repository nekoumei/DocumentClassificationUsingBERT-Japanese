from transformers import BertJapaneseTokenizer, BertForSequenceClassification
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


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    based on: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

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


class DocumentClassifier:
    def __init__(self, max_length=512, batch_size=32, num_labels=2, num_epochs=100, random_seed=None):
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        if random_seed is not None:
            self.seed_everything(random_seed)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
        self.net = BertForSequenceClassification.from_pretrained('bert-base-japanese-whole-word-masking', num_labels=num_labels)
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

    def fit(self, train_df, val_df, early_stopping_rounds=10, fine_tuning_type='fast'):
        logger.info('[Start]Create DataSets from DataFrames')
        train_ds = DataFrameDataset(train_df, fields={'Text': self.TEXT, 'Label': self.LABEL})
        val_ds = DataFrameDataset(val_df, fields={'Text': self.TEXT, 'Label': self.LABEL})
        logger.info('[Finished]Create DataSets from DataFrames')
        self.TEXT.build_vocab(train_ds, min_freq=1)
        self.TEXT.vocab.stoi = self.tokenizer.vocab
        
        logger.info('[Start]Create DataLoaders')
        train_dl = torchtext.data.Iterator(train_ds, batch_size=self.batch_size, train=True)
        val_dl = torchtext.data.Iterator(val_ds, batch_size=self.batch_size, train=False, sort=False)
        logger.info('[Finished]Create DataLoaders')

        dataloaders_dict = {
            'train': train_dl,
            'val': val_dl
        }
        if fine_tuning_type == 'fast':
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in self.net.named_parameters():
                param.requires_grad = False
            # 2. 最後のBertLayerモジュールを勾配計算ありに変更
            for name, param in self.net.bert.encoder.layer[-1].named_parameters():
                param.requires_grad = True
            # 3. 識別器を勾配計算ありに変更
            for name, param in self.net.classifier.named_parameters():
                param.requires_grad = True
            # 最適化手法の設定
            # BERTの元の部分はファインチューニング
            optimizer = optim.Adam([
                {'params': self.net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
                {'params': self.net.classifier.parameters(), 'lr': 5e-5}
            ], betas=(0.9, 0.999))
        elif fine_tuning_type == 'full':
            for name, param in self.net.named_parameters():
                param.requires_grad = True
            # optimの設定
            optimizer = optim.Adam(self.net.parameters(), lr=5e-5, betas=(0.9, 0.999))
        else:
            logger.error('please input fine_tuning_type "fast" or "full"')
            raise ValueError

        # 損失関数の設定
        criterion = nn.CrossEntropyLoss()

        # 学習・検証を実行する。
        self.net = self._train_model(
            self.net, dataloaders_dict, criterion, optimizer, num_epochs=self.num_epochs,
            patience=early_stopping_rounds)

        return self

    @staticmethod
    def _train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, patience):

        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス：{device}")
        logger.info('-----start-------')

        # ネットワークをGPUへ
        net.to(device)

        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        # ミニバッチのサイズ
        batch_size = dataloaders_dict["train"].batch_size

        # early stopping
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # epochのループ
        for epoch in range(num_epochs):
            # epochごとの訓練と検証のループ
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # モデルを訓練モードに
                else:
                    net.eval()   # モデルを検証モードに

                epoch_loss = 0.0  # epochの損失和
                epoch_corrects = 0  # epochの正解数
                iteration = 1

                # 開始時刻を保存
                t_epoch_start = time.time()
                t_iter_start = time.time()
                predictions = []
                ground_truths = []

                # データローダーからミニバッチを取り出すループ
                for batch in (dataloaders_dict[phase]):
                    # batchはTextとLableの辞書型変数

                    # GPUが使えるならGPUにデータを送る
                    inputs = batch.Text[0].to(device)  # 文章
                    labels = batch.Label.to(device)  # ラベル

                    # optimizerを初期化
                    optimizer.zero_grad()

                    # 順伝搬（forward）計算
                    with torch.set_grad_enabled(phase == 'train'):

                        loss, logit = net(input_ids=inputs, labels=labels)                    
                        #loss = criterion(outputs, labels)  # 損失を計算
                        _, preds = torch.max(logit, 1)  # ラベルを予測
                        predictions.append(preds.cpu().numpy())
                        ground_truths.append(labels.data.cpu().numpy())

                        # 訓練時はバックプロパゲーション
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            if (iteration % 1 == 0):  # 10iterに1度、lossを表示
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                acc = (torch.sum(preds == labels.data)
                                    ).double()/batch_size
                                #logger.info('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}'.format(
                                #   iteration, loss.item(), duration, acc))
                                t_iter_start = time.time()

                        iteration += 1

                        # 損失と正解数の合計を更新
                        epoch_loss += loss.item() * batch_size
                        epoch_corrects += torch.sum(preds == labels.data)

                # epochごとのlossと正解率
                t_epoch_finish = time.time()
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double(
                ) / len(dataloaders_dict[phase].dataset)
                if net.num_labels > 2:
                    calc_f1_average = 'macro'
                else:
                    calc_f1_average = 'binary'
                epoch_f1_score = f1_score(np.concatenate(np.array(ground_truths)), np.concatenate(np.array(predictions)), average=calc_f1_average)
                logger.info('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f} F1-Score: {:4f}'.format(epoch+1, num_epochs,
                                                                            phase, epoch_loss, epoch_acc, epoch_f1_score))
                if phase == 'val':
                    early_stopping(epoch_loss, net)
        
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    # load the last checkpoint with the best model
                    net.load_state_dict(torch.load('checkpoint.pt'))
                    return net
        
                t_epoch_start = time.time()

        return net
    
    def predict(self, test_df):
        logger.info('[Start]Create DataSet, DataLoader from DataFrame')
        test_ds = DataFrameDataset(test_df, fields={'Text': self.TEXT})
        test_dl = torchtext.data.Iterator(test_ds, batch_size=self.batch_size, train=False, sort=False)
        logger.info('[Finished]Create DataSet, DataLoader from DataFrame')
        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス：{device}")
        logger.info('-----start-------')
        self.net.eval()
        self.net.to(device)
        logits = []
        for batch in tqdm(test_dl):
            inputs = batch.Text[0].to(device)
            with torch.set_grad_enabled(False):
                logit = self.net(input_ids=inputs)
                logit = F.softmax(logit[0], dim=1).cpu().numpy()
                logits.append(logit)
        logger.info('-----finished-------')
        return np.concatenate(logits, axis=0)