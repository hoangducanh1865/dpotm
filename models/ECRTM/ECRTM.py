import random
import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from utils.config import Config 
from utils import static_utils
from utils.llm import LLM
from transformers import BertModel,BertConfig,DistilBertModel


class ECRTM(nn.Module):
    """
    Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

    Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    """

    def __init__(
        self,
        args,
        vocab,
        vocab_size,
        num_topics=50,
        en_units=200,
        dropout=0.0,
        pretrained_WE=None,
        embed_size=200,
        beta_temp=0.2,
        weight_loss_ECR=100.0,
        sinkhorn_alpha=20.0,
        sinkhorn_max_iter=1000,
        current_run_dir=None,
    ):
        super().__init__()
        self.args = args
        self.is_finetuning = False
        self.device = Config.DEVICE
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.current_run_dir = current_run_dir

        self.beta_ref_path = None
        self.beta_ref = None
        self.topic_word_preference_dataset_path = None
        self.topic_word_preference_dataset = None

        self.theta_ref_path = None
        self.theta_ref = None
        self.doc_topic_preference_dataset_path = None
        self.doc_topic_preference_dataset = None

        # for Jaccard Overlap method
        self.count_drift_topics = 0
        self.beta_prev = None

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(
            torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        )
        self.var2 = nn.Parameter(
            torch.as_tensor(
                (
                    ((1.0 / self.a) * (1 - (2.0 / num_topics))).T
                    + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)
                ).T
            )
        )

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        if args.use_bert_encoder:
            self.encoder=PrefixBERTEncoder(vocab_size=vocab_size,
                                           num_topics=num_topics,
                                           bert_model_name=args.bert_model_name,
                                           prefix_length=args.prefix_length,
                                           freeze_bert=args.freeze_bert,
                                           dropout=dropout)
        else:
            self.fc11 = nn.Linear(vocab_size, en_units)
            self.fc12 = nn.Linear(en_units, en_units)
            self.fc21 = nn.Linear(en_units, num_topics)
            self.fc22 = nn.Linear(en_units, num_topics)
            self.fc1_dropout = nn.Dropout(dropout)

            self.mean_bn = nn.BatchNorm1d(num_topics)
            self.mean_bn.weight.requires_grad = False
            self.logvar_bn = nn.BatchNorm1d(num_topics)
            self.logvar_bn.weight.requires_grad = False
        
        self.theta_dropout = nn.Dropout(dropout)
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False
        
        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(vocab_size, embed_size)
            )
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.weight_loss_ECR = weight_loss_ECR
        self.sinkhorn_alpha = sinkhorn_alpha
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings
        )
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, input):
        if self.args.use_bert_encoder:
            mu,logvar=self.encoder(input_ids=input,
                                   attention_mask=input['attention_mask'])
        else:
            e1 = F.softplus(self.fc11(input))
            e1 = F.softplus(self.fc12(e1))
            e1 = self.fc1_dropout(e1)
            mu = self.mean_bn(self.fc21(e1))
            logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_KL(mu, logvar)

        return theta, loss_KL

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return self.theta_dropout(theta)
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * (
            (var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics
        )
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings
        )
        loss_ECR = self.ECR(cost)
        return loss_ECR

    def load_topic_word_preference_dataset(self):
        self.topic_word_preference_dataset_path = os.path.join(
            self.current_run_dir, "topic_word_preference_dataset.jsonl"
        )
        self.topic_word_preference_dataset = []
        with open(self.topic_word_preference_dataset_path, "r") as f:
            for line in f:
                self.topic_word_preference_dataset.append(line)

        '''print("Loaded topic-word preference dataset")'''

    def load_doc_topic_preference_dataset(self):
        self.doc_topic_preference_dataset_path = os.path.join(
            "data", "doc_topic_preference_dataset", "doc_topic_preference_dataset.jsonl"
        )
        self.doc_topic_preference_dataset = []
        with open(self.doc_topic_preference_dataset_path, "r") as f:
            for line in f:
                self.doc_topic_preference_dataset.append(line)

        '''print("Loaded doc-topic preference dataset")'''

    def load_preference_beta(self):
        # Load reference beta and froze it
        self.beta_ref_path = os.path.join(self.current_run_dir, "beta.npy")
        self.beta_ref = (
            torch.from_numpy(np.load(self.beta_ref_path)).float().to(self.device)
        )
        self.beta_ref.requires_grad = False

    def load_preference_theta(self):
        # Load reference theta and froze it
        self.theta_ref_path = os.path.join(self.current_run_dir, "train_theta.npy")
        self.theta_ref = (
            torch.from_numpy(np.load(self.theta_ref_path)).float().to(self.device)
        )
        self.theta_ref.requires_grad = False

    def get_loss_topic_word_dpo(self, beta, epoch, batch):
        if self.beta_ref is None:
            self.load_preference_beta()

        # Create new preference dataset manually for robustness
        """if epoch % 100 == 1 and batch == 0:
            self.load_topic_word_preference_dataset()
            self.weight_loss_ECR -= 150
            self.ECR = ECR(self.weight_loss_ECR, self.sinkhorn_alpha, self.sinkhorn_max_iter)"""

        if self.args.loss_dpo_topic_word_type == "bradley_terry":

            if self.args.use_jaccard:
                """
                This loop check if at least one word in top-words has just drift, then create a new preference dataset.
                """

                # Detach to prevent gradient tracking
                beta_curr = beta.detach().cpu().numpy()

                if self.beta_prev is not None:
                    """
                    If self.beta_prev is not None then check if half of the topics have drift (at least one top-word changed).
                    If it is, then we will create a new preference dataset.
                    """
                    # TODO
                    # Take current top word indices for k topics
                    _, top_word_indices_list_curr = static_utils.print_topic_words(
                        beta_curr, self.vocab, self.args.num_top_words, False
                    )
                    _, top_word_indices_list_prev = static_utils.print_topic_words(
                        self.beta_prev, self.vocab, self.args.num_top_words, False
                    )

                    # Check drifted topics
                    drift_topics = []
                    for k in range(self.num_topics):
                        set_top_word_indices_curr = set(top_word_indices_list_curr[k])
                        set_top_word_indices_prev = set(top_word_indices_list_prev[k])

                        # Calculate Jaccard Overlap
                        intersection = len(
                            set_top_word_indices_curr.intersection(
                                set_top_word_indices_prev
                            )
                        )
                        union = len(
                            set_top_word_indices_curr.union(set_top_word_indices_prev)
                        )

                        jaccard_ratio = intersection / union

                        # If Jaccard ratio is not 1.0, this topic has drifted
                        if jaccard_ratio < 1.0:
                            drift_topics.append(k)

                    # If there are more than 5/50 topics have drifted, we create a new preference dataset
                    if len(drift_topics) >= 5:
                        self.count_drift_topics += 1
                        if self.count_drift_topics >= 5:
                            self.count_drift_topics = 0
                            llm = LLM(
                                dir_path=self.current_run_dir,
                                num_top_words=self.args.num_top_words,
                            )
                            llm.generate_topic_word_preference_dataset()
                            self.load_topic_word_preference_dataset()

                else:
                    self.beta_prev = beta_curr

            # Indices for preference dataset
            k_indices, w_plus_indices, w_minus_indices = [], [], []

            if self.args.loss_topic_word_dpo_calculation_method == "multiply":

                for line in self.topic_word_preference_dataset:
                    data = json.loads(line)
                    k = data["k"]

                    for w_plus_idx in data["w_plus_indices"]:
                        for w_minus_idx in data["w_minus_indices"]:
                            k_indices.append(k)
                            w_plus_indices.append(w_plus_idx)
                            w_minus_indices.append(w_minus_idx)

            elif self.args.loss_topic_word_dpo_calculation_method == "hard_negative":
                """
                We should use this block since in topic model, there are some cases where some stop words can pass the
                data preprocessing phase, and they get very high beta score -> hard negative words.
                """
                for line in self.topic_word_preference_dataset:
                    data = json.loads(line)
                    k = data["k"]

                    # Find the index of the hardest negative word (the bad word which has highest beta score)
                    hardest_w_minus_idx = -1
                    max_score = -float("inf")

                    # Detach beta score to prevent gradient tracking
                    beta_k_detached = beta[k].detach()

                    for w_minus_idx in data["w_minus_indices"]:
                        score = beta_k_detached[w_minus_idx]

                        if score > max_score:
                            max_score = score
                            hardest_w_minus_idx = w_minus_idx

                    # If there is at least one bad word <=> preference dataset is not None
                    if hardest_w_minus_idx != -1:
                        for w_plus_idx in data["w_plus_indices"]:
                            k_indices.append(k)
                            w_plus_indices.append(w_plus_idx)
                            w_minus_indices.append(hardest_w_minus_idx)

            elif self.args.loss_topic_word_dpo_calculation_method == "hard_positive":
                for line in self.topic_word_preference_dataset:
                    data = json.loads(line)
                    k = data["k"]

                    # Find the index of the hardest positve word (the good word which has loweset beta score)
                    hardest_w_plus_idx = -1
                    min_score = float("inf")

                    # Detach beta score to prevent gradient tracking
                    beta_k_detached = beta[k].detach()

                    for w_plus_idx in data["w_plus_indices"]:
                        score = beta_k_detached[w_plus_idx]

                        if score < min_score:
                            min_score = score
                            hardest_w_plus_idx = w_plus_idx

                    # If there is at least one good word <=> preference dataset is not None
                    if hardest_w_plus_idx != -1:
                        for w_minus_idx in data["w_minus_indices"]:
                            k_indices.append(k)
                            w_plus_indices.append(hardest_w_plus_idx)
                            w_minus_indices.append(w_minus_idx)

            elif self.args.loss_topic_word_dpo_calculation_method == "combined_hard":
                for line in self.topic_word_preference_dataset:
                    data = json.loads(line)
                    k = data["k"]

                    # Find the index of the hardest negative word (the bad word which has highest beta score)
                    hardest_w_minus_idx = -1
                    max_score = -float("inf")

                    # Find the index of the hardest positve word (the good word which has loweset beta score)
                    hardest_w_plus_idx = -1
                    min_score = float("inf")

                    # Detach beta score to prevent gradient tracking
                    beta_k_detached = beta[k].detach()

                    for w_minus_idx in data["w_minus_indices"]:
                        score = beta_k_detached[w_minus_idx]

                        if score > max_score:
                            max_score = score
                            hardest_w_minus_idx = w_minus_idx

                    for w_plus_idx in data["w_plus_indices"]:
                        score = beta_k_detached[w_plus_idx]

                        if score < min_score:
                            min_score = score
                            hardest_w_plus_idx = w_plus_idx

                    # If there is at least one bad word <=> preference dataset is not None
                    if hardest_w_minus_idx != -1:
                        for w_plus_idx in data["w_plus_indices"]:
                            k_indices.append(k)
                            w_plus_indices.append(w_plus_idx)
                            w_minus_indices.append(hardest_w_minus_idx)

                    # If there is at least one good word <=> preference dataset is not None
                    if hardest_w_plus_idx != -1:
                        for w_minus_idx in data["w_minus_indices"]:
                            k_indices.append(k)
                            w_plus_indices.append(hardest_w_plus_idx)
                            w_minus_indices.append(w_minus_idx)

            else:
                raise NotImplementedError("Loss DPO calculation method not supported")

            # If preference data is not None
            if len(k_indices) == 0:
                return torch.tensor(0.0, device=self.device)

            # Convert to tensor for parallel computing
            k_indices = torch.tensor(k_indices, device=self.device, dtype=torch.int64)
            w_plus_indices = torch.tensor(
                w_plus_indices, device=self.device, dtype=torch.int64
            )
            w_minus_indices = torch.tensor(
                w_minus_indices, device=self.device, dtype=torch.int64
            )

            # Calculate delta(s)
            deltas = beta[k_indices, w_plus_indices] - beta[k_indices, w_minus_indices]
            deltas_ref = (
                self.beta_ref[k_indices, w_plus_indices]
                - self.beta_ref[k_indices, w_minus_indices]
            )

            loss_topic_word_dpo = -F.logsigmoid(deltas - deltas_ref).mean()

            return loss_topic_word_dpo

        elif self.args.loss_dpo_topic_word_type == "plackett_luce":
            loss_topic_word_dpo = []

            for line in self.topic_word_preference_dataset:
                data = json.loads(line)
                k = data["k"]
                w_indices = data["w_indices"]

                loss_dpo_per_topic = torch.tensor(1.0, device=self.device)
                for i in range(self.args.num_top_words):
                    denominator = torch.tensor(0.0, device=self.device)
                    for j in range(i, self.args.num_top_words):
                        delta = beta[k][w_indices[j]] - beta[k][w_indices[i]]
                        delta_ref = (
                            self.beta_ref[k][w_indices[j]]
                            - self.beta_ref[k][w_indices[i]]
                        )
                        denominator += torch.exp(delta - delta_ref)

                    loss_dpo_per_topic *= 1.0 / denominator

                loss_topic_word_dpo.append(loss_dpo_per_topic)

            loss_topic_word_dpo = torch.stack(loss_topic_word_dpo)
            loss_topic_word_dpo = -torch.log(loss_topic_word_dpo).mean()

            return loss_topic_word_dpo

        else:
            raise NotImplementedError("Loss DPO type not supported")

    def get_loss_doc_topic_dpo(self, theta, epoch, batch, batch_indices):
        random.seed(epoch + batch)
        if self.theta_ref is None:
            self.load_preference_theta()

        # Create new preference dataset manually for robustness
        '''if epoch % 100 == 1 and batch == 0:
            self.load_doc_topic_preference_dataset()'''

        # Collect all preference pairs
        all_pairs = []
        batch_indices_set = set(batch_indices.cpu().numpy())
        global_to_batch_idx = {
            global_idx.item(): batch_idx
            for batch_idx, global_idx in enumerate(batch_indices)
        }  # @QUESTION: why need .item() here?

        for line in self.doc_topic_preference_dataset:
            data = json.loads(line)
            doc_global_idx = data.get("doc_index", 0)
            if doc_global_idx in batch_indices_set:
                doc_batch_idx = global_to_batch_idx[doc_global_idx]
                ranking = data.get("ranking", data.get("top_5_topics"))
                # Generate all pairwise preferences from ranking
                for i in range(len(ranking)):
                    for j in range(i + 1, len(ranking)):
                        all_pairs.append(
                            (
                                doc_batch_idx,
                                doc_global_idx,
                                ranking[i],
                                ranking[j],
                                ranking,
                            )
                        )
        keep_ratio = 1.0 - self.args.dropout
        num_keep = int(len(all_pairs) * keep_ratio)
        all_pairs = random.sample(all_pairs, num_keep)
        d_indices_batch, d_indices_global, t_plus_indices, t_minus_indices = (
            [],
            [],
            [],
            [],
        )

        if self.args.loss_dpo_doc_topic_type == "bradley_terry":
            """d_indices_batch, t_plus_indices, t_minus_indices = [], [], []
            d_indices_global = [] # For accessing theta_ref

            batch_indices_set = set(batch_indices.cpu().numpy())
            global_to_batch_idx = {global_idx.item(): batch_idx for batch_idx, global_idx in enumerate(batch_indices)}

            # Since number of topics is much smaller number of top words, we use simply Multiply method here to calculate Loss DPO
            for line in self.doc_topic_preference_dataset:
                data = json.loads(line)
                doc_global_idx = data['d']

                # Only process documents which are in current batch
                if doc_global_idx in batch_indices_set:
                    doc_batch_idx = global_to_batch_idx[doc_global_idx]

                    for t_plus_idx in data['t_plus_indices']:
                        for t_minus_idx in data['t_minus_indices']:
                            d_indices_batch.append(doc_batch_idx)
                            d_indices_global.append(doc_global_idx)
                            t_plus_indices.append(t_plus_idx)
                            t_minus_indices.append(t_minus_idx)

            if len(d_indices_batch) == 0:
                return torch.tensor(0.0, device=self.device)

            # Convert to tensor for parallel computing
            d_indices_batch = torch.tensor(d_indices_batch, device=self.device, dtype=torch.int64)
            d_indices_global = torch.tensor(d_indices_global, device=self.device, dtype=torch.int64)
            t_plus_indices = torch.tensor(t_plus_indices, device=self.device, dtype=torch.int64)
            t_minus_indices = torch.tensor(t_minus_indices, device=self.device, dtype=torch.int64)

            # Calculate deltas
            deltas = theta[d_indices_batch, t_plus_indices] - theta[d_indices_batch, t_minus_indices]
            deltas_ref = self.theta_ref[d_indices_global, t_plus_indices] - self.theta_ref[d_indices_global, t_minus_indices]

            loss_doc_topic_dpo = -F.logsigmoid(deltas - deltas_ref).mean()

            return loss_doc_topic_dpo"""
            method=self.args.loss_doc_topic_dpo_calculation_method
            if method == 'multiply':
                # Use all pairs
                for doc_batch_idx, doc_global_idx, t_plus_idx, t_minus_idx, _ in all_pairs:
                    d_indices_batch.append(doc_batch_idx)
                    d_indices_global.append(doc_global_idx)
                    t_plus_indices.append(t_plus_idx)
                    t_minus_indices.append(t_minus_idx)
            elif method == 'hard_negative':
                # For each ranking, pick the hardest negative (lowest theta score)
                for doc_batch_idx, doc_global_idx, _, _, ranking in all_pairs:
                    # Find hardest negative topic for this document
                    theta_doc = theta[doc_batch_idx].detach()
                    hardest_t_minus_idx = min(ranking, key=lambda idx: theta_doc[idx])
                    # For each positive, pair with hardest negative
                    for t_plus_idx in ranking:
                        if t_plus_idx != hardest_t_minus_idx:
                            d_indices_batch.append(doc_batch_idx)
                            d_indices_global.append(doc_global_idx)
                            t_plus_indices.append(t_plus_idx)
                            t_minus_indices.append(hardest_t_minus_idx)
            elif method == 'hard_positive':
                # For each ranking, pick the hardest positive (highest theta score)
                for doc_batch_idx, doc_global_idx, _, _, ranking in all_pairs:
                    theta_doc = theta[doc_batch_idx].detach()
                    hardest_t_plus_idx = max(ranking, key=lambda idx: theta_doc[idx])
                    # For each negative, pair with hardest positive
                    for t_minus_idx in ranking:
                        if t_minus_idx != hardest_t_plus_idx:
                            d_indices_batch.append(doc_batch_idx)
                            d_indices_global.append(doc_global_idx)
                            t_plus_indices.append(hardest_t_plus_idx)
                            t_minus_indices.append(t_minus_idx)
            elif method == 'combined_hard':
                # For each ranking, pair hardest positive with hardest negative
                for doc_batch_idx, doc_global_idx, _, _, ranking in all_pairs:
                    theta_doc = theta[doc_batch_idx].detach()
                    hardest_t_plus_idx = max(ranking, key=lambda idx: theta_doc[idx])
                    hardest_t_minus_idx = min(ranking, key=lambda idx: theta_doc[idx])
                    if hardest_t_plus_idx != hardest_t_minus_idx:
                        d_indices_batch.append(doc_batch_idx)
                        d_indices_global.append(doc_global_idx)
                        t_plus_indices.append(hardest_t_plus_idx)
                        t_minus_indices.append(hardest_t_minus_idx)
            else:
                raise NotImplementedError('Loss DPO calculation method not supported for doc-topic')

            if len(d_indices_batch) == 0:
                return torch.tensor(0.0, device=self.device)

            # Convert to tensor for parallel computing
            d_indices_batch = torch.tensor(d_indices_batch, device=self.device, dtype=torch.int64)
            d_indices_global = torch.tensor(d_indices_global, device=self.device, dtype=torch.int64)
            t_plus_indices = torch.tensor(t_plus_indices, device=self.device, dtype=torch.int64)
            t_minus_indices = torch.tensor(t_minus_indices, device=self.device, dtype=torch.int64)

            # Calculate deltas
            deltas = theta[d_indices_batch, t_plus_indices] - theta[d_indices_batch, t_minus_indices]
            deltas_ref = self.theta_ref[d_indices_global, t_plus_indices] - self.theta_ref[d_indices_global, t_minus_indices]

            loss_doc_topic_dpo = -F.logsigmoid(deltas - deltas_ref).mean()
            return loss_doc_topic_dpo
        elif self.args.loss_dpo_doc_topic_type == "plackett_luce":
            loss_topic_word_dpo = []

            batch_indices_set = set(batch_indices.cpu().numpy())
            global_to_batch_idx = {
                global_idx.item(): batch_idx
                for batch_idx, global_idx in enumerate(batch_indices)
            }

            for line in self.doc_topic_preference_dataset:
                data = json.loads(line)
                # Update field names to match your new format
                doc_global_idx = data["doc_index"]  # Changed from 'd' to 'doc_index'

                # Only process documents which are in current batch
                if doc_global_idx in batch_indices_set:
                    doc_batch_idx = global_to_batch_idx[doc_global_idx]
                    t_indices = data["ranking"]  # Changed from 't_indices' to 'ranking'

                    # Calculate Plackett-Luce probability for this ranking
                    log_prob = torch.tensor(0.0, device=self.device)

                    for i in range(len(t_indices)):
                        # At position i, we need P(t_indices[i] | remaining topics)
                        numerator = (
                            theta[doc_batch_idx, t_indices[i]]
                            - self.theta_ref[doc_global_idx, t_indices[i]]
                        )

                        # Denominator: sum over all remaining topics (positions i to end)
                        denominator_terms = []
                        for j in range(i, len(t_indices)):
                            score_diff = (
                                theta[doc_batch_idx, t_indices[j]]
                                - self.theta_ref[doc_global_idx, t_indices[j]]
                            )
                            denominator_terms.append(score_diff)

                        # Use logsumexp for numerical stability
                        log_denominator = torch.logsumexp(
                            torch.stack(denominator_terms), dim=0
                        )

                        # Add log probability for this position
                        log_prob += numerator - log_denominator

                    # Negate for loss (we want to maximize log probability)
                    loss_topic_word_dpo.append(-log_prob)

            if len(loss_topic_word_dpo) > 0:
                loss_doc_topic_dpo = torch.stack(loss_topic_word_dpo).mean()
            else:
                loss_doc_topic_dpo = torch.tensor(0.0, device=self.device)

            return loss_doc_topic_dpo

        else:
            raise NotImplementedError("Loss DPO type not supported")

    def get_loss_topic_word_regularization(self, beta):
        """beta = self.get_beta()"""
        regularization_term = torch.mean((beta - self.beta_ref) ** 2)
        return regularization_term

    def get_loss_doc_topic_regularization(self, theta, batch_indices):
        theta_ref_batch = self.theta_ref[batch_indices]
        regularization_term = torch.mean((theta - theta_ref_batch) ** 2)
        return regularization_term

    def pairwise_euclidean_distance(self, x, y):
        cost = (
            torch.sum(x**2, axis=1, keepdim=True)
            + torch.sum(y**2, dim=1)
            - 2 * torch.matmul(x, y.t())
        )
        return cost

    def forward(self, input, epoch, batch):
        bow = input["data"]
        theta, loss_KL = self.encode(bow if not self.args.use_bert_encoder else input['input_ids'])
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()

        if not self.is_finetuning:
            loss = loss_TM + loss_ECR

            rst_dict = {"loss": loss, "loss_TM": loss_TM, "loss_ECR": loss_ECR}

        else:
            loss = loss_TM + loss_ECR
            if self.args.finetune_beta:
                loss_topic_word_dpo = self.get_loss_topic_word_dpo(beta, epoch, batch)
                loss_topic_word_reg = self.get_loss_topic_word_regularization(beta)
                loss += (
                    self.args.weight_topic_word_dpo * loss_topic_word_dpo + self.args.weight_topic_word_reg * loss_topic_word_reg
                )

            batch_indices = input["indices"]

            if self.args.finetune_theta:
                loss_doc_topic_dpo = self.get_loss_doc_topic_dpo(
                    theta, epoch, batch, batch_indices
                )
                loss_doc_topic_reg = self.get_loss_doc_topic_regularization(
                    theta, batch_indices
                )
                loss += (
                    self.args.weight_doc_topic_dpo * loss_doc_topic_dpo + self.args.weight_doc_topic_reg * loss_doc_topic_reg
                )

            """if self.args.finetune_beta and self.args.finetune_theta:
                print(f'Epoch: {epoch} - Batch: {batch} - Loss TM: {loss_TM} - Loss ECR: {loss_ECR} - Loss topic-word DPO: {loss_topic_word_dpo} - Loss topic-word Reg: {loss_topic_word_reg} - Loss doc-topic DPO: {loss_doc_topic_dpo} - Loss doc-topic Reg: {loss_doc_topic_reg}')
            elif self.args.finetune_beta:
                print(f'Epoch: {epoch} - Batch: {batch} - Loss TM: {loss_TM} - Loss ECR: {loss_ECR} - Loss topic-word DPO: {loss_topic_word_dpo} - Loss topic-word Reg: {loss_topic_word_reg}')
            elif self.args.finetune_theta:
                print(f'Epoch: {epoch} - Batch: {batch} - Loss TM: {loss_TM} - Loss ECR: {loss_ECR} - Loss doc-topic DPO: {loss_doc_topic_dpo} - Loss doc-topic Reg: {loss_doc_topic_reg}')"""

            if self.args.finetune_beta and self.args.finetune_theta:
                rst_dict = {
                    "loss": loss,
                    "loss_TM": loss_TM,
                    "loss_ECR": loss_ECR,
                    "loss_topic_word_dpo": loss_topic_word_dpo,
                    "loss_topic_word_reg": loss_topic_word_reg,
                    "loss_doc_topic_dpo": loss_doc_topic_dpo,
                    "loss_doc_topic_reg": loss_doc_topic_reg
                }
            elif self.args.finetune_beta:
                rst_dict = {
                    "loss": loss,
                    "loss_TM": loss_TM,
                    "loss_ECR": loss_ECR,
                    "loss_topic_word_dpo": loss_topic_word_dpo,
                    "loss_topic_word_reg": loss_topic_word_reg
                }
            elif self.args.finetune_theta:
                rst_dict = {
                    "loss": loss,
                    "loss_TM": loss_TM,
                    "loss_ECR": loss_ECR,
                    "loss_doc_topic_dpo": loss_doc_topic_dpo,
                    "loss_doc_topic_reg": loss_doc_topic_reg
                }

        return rst_dict
class PrefixBERTEncoder(nn.Module):
    def __init__(self,vocab_size,num_topics,dropout,bert_model_name,prefix_length,freeze_bert):
        super().__init__()
        self.vocab_size=vocab_size
        self.num_topics=num_topics
        self.dropout=nn.Dropout(dropout)
        self.bert_model_name=bert_model_name
        self.prefix_length=prefix_length
        self.freeze_bert=freeze_bert
        if 'distilbert' in bert_model_name.lower():
            self.bert=DistilBertModel.from_pretrained(bert_model_name)
        else:
            self.bert=BertModel.from_pretrained(bert_model_name)
        self.hidden_size=self.bert.config.hidden_size
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad=False
        self.prefix_tokens=nn.Parameter(
            torch.randn(prefix_length,self.hidden_size)
        )
        self.fc_mu=nn.Linear(self.hidden_size,num_topics)
        self.fc_logvar=nn.Linear(self.hidden_size,num_topics)
        self.mean_bn=nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad=False
        self.logvar_bn=nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad=False
    def forward(self,input_ids,attention_mask):
        """

        Args:
            input_ids: shape is (batch_size, seq_len) - tokenized input
            attention_mask: shape is (batch_size, seq_len) - attention mask
        
        Returns:
            mu, logvar
        """
        batch_size=input_ids.size(0)
        
        # Expand prefix for batch
        prefix_embeddings=self.prefix_tokens.unsqueeze(0).expand(batch_size,-1,-1) # Shape is (batch_size, prefix_length, hidden_size) 
        
        bert_outputs=self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               return_dict=True) # Get BERT embedding
        
        # @QUESTION 
        # Use [CLS] token representation
        cls_output=bert_outputs.last_hidden_state[:,0,:] # (batch_size, hidden_size)
        
        # @QUESTION
        # Add prefix information to CLS token
        prefix_mean=prefix_embeddings.mean(dim=1) # (batch_size, hidden_size) 
        combined=cls_output+prefix_mean
        combined=self.dropout(combined)
        
        mu=self.mean_bn(self.fc_mu(combined))
        logvar=self.logvar_bn(self.fc_logvar(combined))
        return mu,logvar
        
        