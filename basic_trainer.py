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


class BasicTrainer:
    def __init__(
        self,
        model,
        epochs,
        learning_rate=0.002,
        batch_size=200,
        use_lr_scheduler=None,
        lr_step_size=125,
        log_interval=5,
        device="cuda",
        args=None,
        checkpoint_dir=None,
        llm=None,
        dataset=None,
        current_run_dir=None,
    ):
        self.args = args
        self.model = model
        self.epochs = epochs
        self.finetune_epochs = args.finetune_epochs
        self.finetune_beta = args.finetune_beta
        self.finetune_theta = args.finetune_theta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = self.make_optimizer()
        self.llm = llm
        if use_lr_scheduler:
            self.lr_scheduler = self.make_lr_scheduler()
        self.dataset = dataset
        self.current_run_dir = current_run_dir
        self.logger = logging.getLogger("main")

    def make_optimizer(self):
        args_dict = {
            "params": self.model.parameters(),
            "lr": self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self):
        lr_scheduler = StepLR(
            self.optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False
        )

        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data if not self.args.use_bert_encoder else dataset_handler.train_dataloader)

        return top_words, train_theta

    def train(
        self, dataset_handler, start_epoch, end_epoch, verbose=False, evaluate_fn=None
    ):
        """
        Combined training method that handles both regular training and DPO fine-tuning
        """
        data_size = len(dataset_handler.train_dataloader.dataset)

        # Regular training phase
        for epoch in tqdm(range(start_epoch, end_epoch + 1)):

            self.model.train()
            loss_rst_dict = defaultdict(float)
            # wandb.log({'epoch': epoch})

            for batch, batch_data in enumerate(dataset_handler.train_dataloader):
                '''batch_size = len(batch_data["data"])

                start_idx = batch * dataset_handler.train_dataloader.batch_size
                actual_batch_size = min(batch_size, data_size - start_idx)
                end_idx = start_idx + actual_batch_size

                batch_indices = torch.arange(start_idx, end_idx, device=self.device)
                batch_data["indices"] = batch_indices'''
                for key in batch_data:
                    if isinstance(batch_data[key],torch.Tensor):
                        batch_data[key]=batch_data[key].to(self.device)

                self.optimizer.zero_grad()
                rst_dict = self.model(batch_data, epoch, batch)
                batch_loss = rst_dict["loss"]
                batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                self.optimizer.step()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data["data"])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            # for key in loss_rst_dict:
            # wandb.log({key: loss_rst_dict[key] / data_size})

            self.lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f"Epoch: {epoch:03d}"
                for key in loss_rst_dict:
                    output_log += f" {key}: {loss_rst_dict[key] / data_size :.3f}"

                print(output_log)
                self.logger.info(output_log)

            # Evaluate model every 100 epochs during fine-tuning phase and do not evaluate at the end of fine-tuning phase since we will evaluate in main
            if (
                epoch > self.args.epochs
                and epoch % 100 == 0
                and epoch != (self.args.epochs + self.args.finetune_epochs)
            ):
                print("=" * 32)
                self.logger.info(f"Evaluation at epoch {epoch}")
                evaluate_fn(epoch)

            if epoch >= self.epochs and epoch % 100 == 0:
                self.save_checkpoint(epoch)

    def test(self,input_data):
        theta = list()
        
        if self.args.use_bert_encoder:
            with torch.no_grad():
                self.model.eval()
                for batch_data in input_data:
                    # Move all tensors in batch_data to the correct device
                    for key in batch_data:
                        if isinstance(batch_data[key], torch.Tensor):
                            batch_data[key] = batch_data[key].to(self.device)
                    
                    batch_theta = self.model.get_theta(batch_data)
                    theta.extend(batch_theta.cpu().tolist())
        else:
            data_size = input_data.shape[0]
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

    def export_top_words(self, vocab, num_top_words, print_topic=True):
        beta = self.export_beta()
        top_words, top_word_indices = static_utils.print_topic_words(
            beta, vocab, num_top_words, print_topic
        )
        return top_words, top_word_indices

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data if not self.args.use_bert_encoder else dataset_handler.train_dataloader)
        test_theta = self.test(dataset_handler.test_data if not self.args.use_bert_encoder else dataset_handler.test_dataloader)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, "beta.npy"), beta)
        return beta

    def save_top_words(
        self, vocab, num_top_words, dir_path, suffix="", print_topic=False
    ):
        top_words, top_word_indices = self.export_top_words(
            vocab, num_top_words, print_topic
        )

        with open(
            os.path.join(dir_path, suffix + f"top_words_{num_top_words}.txt"), "w"
        ) as f:
            for i, words in enumerate(top_words):
                f.write(words + "\n")

        with open(
            os.path.join(dir_path, suffix + f"top_words_{num_top_words}.jsonl"), "w"
        ) as f:
            for k, (words, indices) in enumerate(zip(top_words, top_word_indices)):
                words_list = words.split()
                top_words_with_indices = []
                for word, idx in zip(words_list, indices):
                    top_words_with_indices.append({word: idx})

                topic_data = {"k": k, "top_words": top_words_with_indices}

                f.write(json.dumps(topic_data) + "\n")

        # top_words = " ".join(top_words)  # Removed - keep as list
        return top_words

    def save_theta(self, dataset_handler, dir_path, log_theta_predictions=True):
        train_theta, test_theta = self.export_theta(dataset_handler)
        if log_theta_predictions:
            # @TODO: print label for each train document
            self.logger.info("Predictions on train set:")
            preds = np.argmax(train_theta, axis=1)
            pred_str = " ".join(str(pred) for pred in preds)
            self.logger.info(pred_str)

            # @TODO: print label for each test document
            """self.logger.info('Predictions on test set:')
            preds = np.argmax(test_theta, axis=1)
            pred_str = ' '.join(str(pred) for pred in preds)
            self.logger.info(pred_str)"""

        np.save(os.path.join(dir_path, "train_theta.npy"), train_theta)
        np.save(os.path.join(dir_path, "test_theta.npy"), test_theta)

        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, "train_argmax_theta.npy"), train_argmax_theta)
        np.save(os.path.join(dir_path, "test_argmax_theta.npy"), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, "word_embeddings"):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, "word_embeddings.npy"), word_embeddings)
            self.logger.info(f"word_embeddings size: {word_embeddings.shape}")

        if hasattr(self.model, "topic_embeddings"):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, "topic_embeddings.npy"), topic_embeddings)
            self.logger.info(f"topic_embeddings size: {topic_embeddings.shape}")

            topic_dist = scipy.spatial.distance.cdist(
                topic_embeddings, topic_embeddings
            )
            np.save(os.path.join(dir_path, "topic_dist.npy"), topic_dist)

        if hasattr(self.model, "group_embeddings"):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, "group_embeddings.npy"), group_embeddings)
            self.logger.info(f"group_embeddings size: {group_embeddings.shape}")

            group_dist = scipy.spatial.distance.cdist(
                group_embeddings, group_embeddings
            )
            np.save(os.path.join(dir_path, "group_dist.npy"), group_dist)

        return word_embeddings, topic_embeddings

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
        )

        torch.save(checkpoint, checkpoint_path)

        self.logger.info(f"Checkpint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        self.logger.info(
            f"Checkpoint loaded: {checkpoint_path}, resuming at epoch {start_epoch}"
        )

        return start_epoch
