import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy
from time import time


class DPOFinetuner:
    def __init__(self, model, epochs=500, finetune_epochs=100, learning_rate=0.002, batch_size=200, use_lr_scheduler=None, lr_step_size=125, log_interval=5, device="cuda", checkpoint_dir=None, preference_dataset_path=None):
        self.model = model
        self.epochs = epochs
        self.finetune_epochs = finetune_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.preference_dataset_path = preference_dataset_path
        self.optimizer = self.make_optimizer()
        if use_lr_scheduler:
            self.lr_scheduler = self.make_lr_scheduler()

        self.logger = logging.getLogger('main')

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self):
        lr_scheduler = StepLR(
            self.optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def finetune(self, dataset_handler, verbose=False):
        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(self.epochs + 1, self.epochs + self.finetune_epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            # wandb.log({'epoch': epoch})

            for batch_data in dataset_handler.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                self.optimizer.zero_grad()
                batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                self.optimizer.step()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            # for key in loss_rst_dict:
                # wandb.log({key: loss_rst_dict[key] / data_size})
            
            self.lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)
                self.logger.info(output_log)
            
            if epoch == 400 or epoch == 500:
                self.save_checkpoint(epoch)

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words):
        beta = self.export_beta()
        top_words, top_word_indices = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words, top_word_indices

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        test_theta = self.test(dataset_handler.test_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words, top_word_indices = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'finetuned_top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
    
        with open(os.path.join(dir_path, f'finetuned_top_words_{num_top_words}.jsonl'), 'w') as f:
            for k, (words, indices) in enumerate(zip(top_words, top_word_indices)):
                words_list = words.split()
                top_words_with_indices = []
                for word, idx in zip(words_list, indices):
                    top_words_with_indices.append({word: idx})
                
                topic_data = {
                    'k': k,
                    'top_words': top_words_with_indices
                }
                
                f.write(json.dumps(topic_data) + '\n')
        
        top_words = ' '.join(top_words)
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        
        self.logger.info(f'Checkpoint loaded: {checkpoint_path}, resuming at epoch {start_epoch}')
        
        return start_epoch