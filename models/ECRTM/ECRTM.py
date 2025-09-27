import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from utils.configs import Configs as cfg
from utils import static_utils
from utils.preference_dataset_creator import PreferenceDatasetCreator
import json


class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, args, vocab, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2, weight_loss_ECR=100.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=1000, current_run_dir=None):
        super().__init__()

        self.is_finetuning = False
        self.device = cfg.DEVICE
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_top_words = args.num_top_words
        self.beta_temp = beta_temp
        self.current_run_dir = current_run_dir
        
        self.beta_ref_path = None
        self.beta_ref = None
        self.preference_dataset_path = None
        self.preference_dataset = None
        
        self.weight_dpo = args.weight_dpo
        self.weight_reg = args.weight_reg
        
        # Methods to calculate DPO loss
        self.loss_dpo_calculation_method = args.loss_dpo_calculation_method
        self.use_jaccard = args.use_jaccard
        self.loss_dpo_type = args.loss_dpo_type
        self.count_drift_topics = 0
        
        # for Jaccard Overlap method
        self.beta_prev = None

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
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
            return theta, loss_KL
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR
    
    def load_preference_dataset(self):
        self.preference_dataset_path = os.path.join(self.current_run_dir, 'preference_dataset.jsonl')
        if self.preference_dataset is None:
            self.preference_dataset = []
            with open(self.preference_dataset_path, 'r') as f:
                for line in f:
                    self.preference_dataset.append(line)
        
        # Also create reference beta and frozen it
        self.beta_ref_path = os.path.join(self.current_run_dir, 'beta.npy')
        self.beta_ref = torch.from_numpy(np.load(self.beta_ref_path)).float().to(self.device)
        self.beta_ref.requires_grad = False
    
    def get_loss_dpo(self):
        if self.preference_dataset is None:
            self.load_preference_dataset()
                
        beta = self.get_beta()
            
        if self.loss_dpo_type == 'bradley_terry':
            
            if self.use_jaccard == True:
                '''
                This loop check if at least one word in top-words has just drift, then create a new preference dataset.
                '''
                
                # Detach to prevent gradient tracking
                beta_curr = beta.detach().cpu().numpy()
                
                if self.beta_prev is not None:
                    '''
                    If self.beta_prev is not None then check if half of the topics have drift (at least one top-word changed).
                    If it is, then we will create a new preference dataset.
                    '''
                    # TODO
                    # Take current top word indices for k topics 
                    _, top_word_indices_list_curr = static_utils.print_topic_words(beta_curr, self.vocab, self.num_top_words, False)
                    _, top_word_indices_list_prev = static_utils.print_topic_words(self.beta_prev, self.vocab, self.num_top_words, False)
                    
                    # Check drifted topics
                    drift_topics = []
                    for k in range(self.num_topics):
                        set_top_word_indices_curr = set(top_word_indices_list_curr[k])
                        set_top_word_indices_prev = set(top_word_indices_list_prev[k])
                        
                        # Calculate Jaccard Overlap
                        intersection = len(set_top_word_indices_curr.intersection(set_top_word_indices_prev))
                        union = len(set_top_word_indices_curr.union(set_top_word_indices_prev))
                        
                        jaccard_ratio = intersection / union
                        
                        # If Jaccard ratio is not 1.0, this topic has drifted
                        if jaccard_ratio < 1.0:
                            drift_topics.append(k)
                    
                    # If there are more than 5/50 topics have drifted, we create a new preference dataset
                    if len(drift_topics) >= 5:
                        self.count_drift_topics += 1
                        if self.count_drift_topics >= 5:
                            self.count_drift_topics = 0
                            preference_dataset_creator = PreferenceDatasetCreator(dir_path=self.current_run_dir, num_top_words=self.num_top_words)
                            preference_dataset_creator.create()
                            self.load_preference_dataset()
                    
                else:
                    self.beta_prev = beta_curr
            
            # Indices for preference dataset
            k_indices, w_plus_indices, w_minus_indices = [], [], []
            
            if self.loss_dpo_calculation_method == 'multiply':
                
                for line in self.preference_dataset:
                    data = json.loads(line)
                    k = data['k']
                    
                    for w_plus_idx in data['w_plus_indices']:
                        for w_minus_idx in data['w_minus_indices']:
                            k_indices.append(k)
                            w_plus_indices.append(w_plus_idx)
                            w_minus_indices.append(w_minus_idx)
            
            elif self.loss_dpo_calculation_method == 'hard_negative':
                '''
                We should use this block since in topic model, there are some cases where some stop words can pass the 
                data preprocessing phase, and they get very high beta score -> hard negative words.
                '''
                for line in self.preference_dataset:
                    data = json.loads(line)
                    k = data['k']
                    
                    # Find the index of the hardest negative word (the bad word which has highest beta score)
                    hardest_w_minus_idx = -1
                    max_score = -float('inf')
                    
                    # Detach beta score to prevent gradient tracking
                    beta_k_detached = beta[k].detach()
                    
                    for w_minus_idx in data['w_minus_indices']:
                        score = beta_k_detached[w_minus_idx]
                        
                        if score > max_score:
                            max_score = score
                            hardest_w_minus_idx = w_minus_idx
                    
                    # If there is at least one bad word <=> preference dataset is not None
                    if hardest_w_minus_idx != -1:
                        for w_plus_idx in data['w_plus_indices']:
                            k_indices.append(k)
                            w_plus_indices.append(w_plus_idx)
                            w_minus_indices.append(hardest_w_minus_idx)
            
            elif self.loss_dpo_calculation_method == 'hard_positive':
                for line in self.preference_dataset:
                    data = json.loads(line)
                    k = data['k']
                    
                    # Find the index of the hardest positve word (the good word which has loweset beta score)
                    hardest_w_plus_idx = -1
                    min_score = float('inf')
                    
                    # Detach beta score to prevent gradient tracking
                    beta_k_detached = beta[k].detach()
                    
                    for w_plus_idx in data['w_plus_indices']:
                        score = beta_k_detached[w_plus_idx]
                        
                        if score < min_score:
                            min_score = score
                            hardest_w_plus_idx = w_plus_idx
                    
                    # If there is at least one good word <=> preference dataset is not None
                    if hardest_w_plus_idx != -1:
                        for w_minus_idx in data['w_minus_indices']:
                            k_indices.append(k)
                            w_plus_indices.append(hardest_w_plus_idx)
                            w_minus_indices.append(w_minus_idx)
            
            elif self.loss_dpo_calculation_method == 'combined_hard':
                for line in self.preference_dataset:
                    data = json.loads(line)
                    k = data['k']
                    
                    # Find the index of the hardest negative word (the bad word which has highest beta score)
                    hardest_w_minus_idx = -1
                    max_score = -float('inf')
                    
                    # Find the index of the hardest positve word (the good word which has loweset beta score)
                    hardest_w_plus_idx = -1
                    min_score = float('inf')
                    
                    # Detach beta score to prevent gradient tracking
                    beta_k_detached = beta[k].detach()
                    
                    for w_minus_idx in data['w_minus_indices']:
                        score = beta_k_detached[w_minus_idx]
                        
                        if score > max_score:
                            max_score = score
                            hardest_w_minus_idx = w_minus_idx
                    
                    for w_plus_idx in data['w_plus_indices']:
                        score = beta_k_detached[w_plus_idx]
                        
                        if score < min_score:
                            min_score = score
                            hardest_w_plus_idx = w_plus_idx
                            
                    # If there is at least one bad word <=> preference dataset is not None
                    if hardest_w_minus_idx != -1:
                        for w_plus_idx in data['w_plus_indices']:
                            k_indices.append(k)
                            w_plus_indices.append(w_plus_idx)
                            w_minus_indices.append(hardest_w_minus_idx)
                    
                    # If there is at least one good word <=> preference dataset is not None
                    if hardest_w_plus_idx != -1:
                        for w_minus_idx in data['w_minus_indices']:
                            k_indices.append(k)
                            w_plus_indices.append(hardest_w_plus_idx)
                            w_minus_indices.append(w_minus_idx)
                
            else:
                raise NotImplementedError('Loss DPO calculation method not supported')   
                            
            # If preference data is not None
            if len(k_indices) == 0:
                return torch.tensor(0.0, device=self.device)
                
            # Convert to tensor for parallel computing
            k_indices = torch.tensor(k_indices, device=self.device, dtype=torch.int64)
            w_plus_indices = torch.tensor(w_plus_indices, device=self.device, dtype=torch.int64)
            w_minus_indices = torch.tensor(w_minus_indices, device=self.device, dtype=torch.int64)
            
            # Calculate delta(s)
            deltas = beta[k_indices, w_plus_indices] - beta[k_indices, w_minus_indices]
            deltas_ref = self.beta_ref[k_indices, w_plus_indices] - self.beta_ref[k_indices, w_minus_indices]
            
            loss_dpo = -F.logsigmoid(deltas - deltas_ref).mean()
            
            return loss_dpo

        elif self.loss_dpo_type == 'plackett_luce':
            loss_dpo = []
            
            for line in self.preference_dataset:
                data = json.loads(line)
                k = data['k']
                w_indices = data['w_indices']
                
                loss_dpo_per_topic = torch.tensor(1.0, device=self.device)
                for i in range(self.num_top_words):
                    denominator = torch.tensor(0.0, device=self.device)
                    for j in range(i, self.num_top_words):
                        delta = beta[k][w_indices[j]] - beta[k][w_indices[i]]
                        delta_ref = self.beta_ref[k][w_indices[j]] - self.beta_ref[k][w_indices[i]]
                        denominator += torch.exp(delta - delta_ref)
                        
                    loss_dpo_per_topic *= 1.0 / denominator
                    
                loss_dpo.append(loss_dpo_per_topic)
            
            loss_dpo = torch.stack(loss_dpo)
            loss_dpo = -torch.log(loss_dpo).mean()
                        
            return loss_dpo
            

    def get_loss_regularization(self):
        beta = self.get_beta()
        regularization_term = torch.mean((beta - self.beta_ref) ** 2)
        return regularization_term

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, input):
        bow = input["data"]
        theta, loss_KL = self.encode(input['data'])
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        
        if not self.is_finetuning:
            
            loss = loss_TM + loss_ECR

            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR
            }
        
        else:
            loss_DPO = self.get_loss_dpo()
            
            loss_regularization = self.get_loss_regularization()
            
            loss = loss_TM + loss_ECR + self.weight_dpo * loss_DPO + self.weight_reg * loss_regularization

            rst_dict = {
                'loss': loss,
                'loss_TM': loss_TM,
                'loss_ECR': loss_ECR,
                'loss_DPO': loss_DPO,
                'loss_regularization': loss_regularization
            }

        return rst_dict
        