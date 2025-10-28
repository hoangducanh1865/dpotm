import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse
import scipy.io
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from . import file_utils
import os


def load_contextual_embed(texts, device, model_name="all-mpnet-base-v2", show_progress_bar=True):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return embeddings


class DatasetHandler(Dataset):
    def __init__(self, args, data, contextual_embed=None, texts=None):
        self.args=args
        self.data = data
        self.texts=texts
        self.contextual_embed = None
        if contextual_embed is not None:
            assert data.shape[0] == contextual_embed.shape[0], "Data and contextual embeddings should have the same number of samples"
            self.contextual_embed = contextual_embed
        if args.use_bert_encoder:
            self.tokenizer=BertTokenizer.from_pretrained(args.bert_model_name)

    def __len__(self):
        # Update this according to your data size
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        item={
            'data':self.data[idx],
            'indices':idx
        }

        if self.contextual_embed is not None:
            item['contextual_embed']=self.contextual_embed[idx]

        if self.args.use_bert_encoder:
            encoding=self.tokenizer(self.texts[idx],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=512,
                                    return_tensors='pt')
            item['input_ids']=encoding['input_ids'].squeeze(0)
            item['attention_mask']=encoding['attention_mask'].squeeze(0)
        return item


class RawDatasetHandler:
    def __init__(self, docs, preprocessing, batch_size=200, device='cpu', as_tensor=False, contextual_embed=False):

        rst = preprocessing.preprocess(docs)
        self.train_data = rst['train_bow']
        self.train_texts = rst['train_texts']
        self.vocab = rst['vocab']

        self.vocab_size = len(self.vocab)

        if contextual_embed:
            self.train_contextual_embed = load_contextual_embed(
                self.train_texts, device)
            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if contextual_embed:
                self.train_data = np.concatenate(
                    (self.train_data, self.train_contextual_embed), axis=1)

            self.train_data = torch.from_numpy(
                self.train_data).float().to(device)
            self.train_dataloader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True)


class BasicDatasetHandler:
    def __init__(self, args, dataset_dir, batch_size=200, read_labels=False, device='cpu', as_tensor=False, contextual_embed=False, plm_model="all-mpnet-base-v2"):
        # train_bow: NxV
        # test_bow: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.
        self.args=args
        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)
        self.plm_model = plm_model

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(
            self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if contextual_embed:
            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'train_bert.npz')):
                self.train_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'train_bert.npz'))['arr_0']
            else:
                self.train_contextual_embed = load_contextual_embed(
                    self.train_texts, device, model_name=self.plm_model)

            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'test_bert.npz')):
                self.test_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'test_bert.npz'))['arr_0']
            else:
                self.test_contextual_embed = load_contextual_embed(
                    self.test_texts, device, model_name=self.plm_model)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            # if not contextual_embed:  # to be fixed with an additional argument
            #     self.train_data = self.train_bow
            #     self.test_data = self.test_bow
            # else:
            #     self.train_data = np.concatenate((self.train_bow, self.train_contextual_embed), axis=1)
            #     self.test_data = np.concatenate((self.test_bow, self.test_contextual_embed), axis=1)
            self.train_data = self.train_bow
            self.test_data = self.test_bow

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)

            if contextual_embed:

                self.train_contextual_embed = torch.from_numpy(
                    self.train_contextual_embed).to(device)
                self.test_contextual_embed = torch.from_numpy(
                    self.test_contextual_embed).to(device)

                train_dataset = DatasetHandler(
                    args,
                    self.train_data, self.train_contextual_embed,
                    texts=self.train_texts)
                test_dataset = DatasetHandler(
                    args,
                    self.test_data, self.test_contextual_embed,
                    texts=self.test_texts)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)

            else:
                train_dataset = DatasetHandler(args,
                                               self.train_data,
                                               texts=self.train_texts)
                test_dataset = DatasetHandler(args,
                                              self.test_data,
                                              texts=self.train_texts)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):
        
        # Try different file naming conventions
        try:
            self.train_bow = scipy.sparse.load_npz(
                f'{path}/train_bow.npz').toarray().astype('float32')
        except FileNotFoundError:
            self.train_bow = scipy.sparse.load_npz(
                f'{path}/bow.npz').toarray().astype('float32')
            
        self.test_bow = scipy.sparse.load_npz(
            f'{path}/test_bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(
            f'{path}/word_embeddings.npz').toarray().astype('float32')

        try:
            self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        except FileNotFoundError:
            self.train_texts = file_utils.read_text(f'{path}/texts.txt')
            
        try:
            self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')
        except FileNotFoundError:
            self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        if read_labels:
            try:
                self.train_labels = np.loadtxt(
                    f'{path}/train_labels.txt', dtype=int)
            except FileNotFoundError:
                self.train_labels = np.loadtxt(
                    f'{path}/labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')
