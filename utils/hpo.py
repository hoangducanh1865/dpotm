import os
import copy
import numpy as np
import evaluations
import basic_trainer
from models.ECRTM import ECRTM
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from collections import defaultdict
from models.ECRTM import ECRTM
from utils.llm import LLM


class HPO:
    @staticmethod
    def ecrtm_finetune_objective_fn(args,dataset,current_run_dir,pretrainWE):
        def objective(config):
            weight_doc_topic_dpo=config.get('weight_doc_topic_dpo',args.weight_doc_topic_dpo)
            weight_doc_topic_reg=config.get('weight_doc_topic_reg',args.weight_doc_topic_reg)
            finetune_epochs=int(round(config.get('finetune_epochs',args.finetune_epochs)))
            args_copy=copy.deepcopy(args)
            args_copy.weight_doc_topic_dpo=weight_doc_topic_dpo
            args_copy.weight_doc_topic_reg=weight_doc_topic_reg
            model=ECRTM(args_copy,
                       vocab = dataset.vocab,
                       vocab_size=dataset.vocab_size, 
                       num_topics=args.num_topics, 
                       dropout=args.dropout, 
                       pretrained_WE=pretrainWE if args.use_pretrainWE else None, 
                       weight_loss_ECR=args.weight_ECR,
                       current_run_dir=current_run_dir).to(args.device)
            trainer = basic_trainer.BasicTrainer(model, epochs=args.epochs,
                                                learning_rate=args.lr,
                                                batch_size=args.batch_size,
                                                use_lr_scheduler=args.lr_scheduler,
                                                lr_step_size=args.lr_step_size,
                                                device=args.device,
                                                args=args,
                                                checkpoint_dir=None,
                                                dataset=dataset,
                                                current_run_dir=current_run_dir)
            if args.checkpoint_path:
                start_epoch=trainer.load_checkpoint(args.checkpoint_path)
            else:
                checkpoint_path=os.path.join(current_run_dir,'checkpoints',f'checkpoint_epoch_{args.epochs}')
                start_epoch=trainer.load_checkpoint(checkpoint_path)
            train_theta_pre=trainer.test(dataset.train_data)
            test_theta_pre=trainer.test(dataset.test_data)
            nmi_pre=0.0
            purity_pre=0.0
            clustering_results_pre=evaluations.evaluate_clustering(test_theta_pre,dataset.test_labels)
            nmi_pre=clustering_results_pre['NMI']
            purity_pre=clustering_results_pre['Purity']
            
            model.is_finetuning=True
            trainer.llm=LLM(current_run_dir,args.num_top_words,dataset.vocab,args_copy)
            trainer.train(dataset,start_epoch,start_epoch-1+finetune_epochs,evaluate_fn=None) # During HPO, we just care about loss objective
            train_theta_post=trainer.test(dataset.train_data)
            test_theta_post=trainer.test(dataset.test_data)
            nmi_post=0.0
            purity_post=0.0
            clustering_results_post=evaluations.evaluate_clustering(test_theta_post,dataset.test_labels)
            nmi_post=clustering_results_post['NMI']
            purity_post=clustering_results_post['Purity']
            
            nmi_improvement=nmi_post-nmi_pre
            purity_improvement=purity_post-purity_pre
            print(f'NMI improvement: {nmi_improvement:.4f} ({nmi_pre:.4f} -> {nmi_post:.4f})')
            print(f'Purity improvement: {purity_improvement:.4f} ({purity_pre:.4f} -> {purity_post:.4f})')
            score=0.5*nmi_improvement+0.5*purity_improvement
            print(f'Score: {score:.4f}')
            return -score
        return objective
    @staticmethod
    def bayesian_search(args, dataset, current_run_dir, pretrainWE, config_space, initial_config, 
                       num_trials=20, n_random_init=5):
        """
        Run Bayesian Optimization for ECRTM fine-tuning hyperparameters.
        
        Args:
            num_trials: Total number of HPO trials
            n_random_init: Number of random trials before starting BO
        """
        print(f"\n{'='*32}")
        print(f"Starting Bayesian Optimization for ECRTM Fine-tuning")
        print(f"{'='*32}")
        print(f"Total trials: {num_trials}")
        print(f"Random initialization: {n_random_init} trials")
        print(f"Bayesian optimization: {num_trials - n_random_init} trials")
        print(f"Search space: {list(config_space.keys())}")
        print(f"Initial config: {initial_config}")
        
        # Create searcher and scheduler
        searcher = BayesianSearcher(config_space, initial_config, n_random_init=n_random_init)
        scheduler = BasicScheduler(searcher)
        
        # Create objective function
        objective_fn = HPO.ecrtm_finetune_objective_fn(
            args, dataset, current_run_dir, pretrainWE
        )
        
        # Run HPO
        tuner = HPOTuner(scheduler=scheduler, objective_fn=objective_fn)
        tuner.run(number_of_trials=num_trials)
        
        # Get best config
        best_config, best_score = tuner.get_best_config()
        
        print(f"\n{'='*32}")
        print("Bayesian Optimization Results")
        print(f"{'='*32}")
        print(f"Best config:")
        for key, value in best_config.items():
            print(f"  {key}: {value:.4f}")
        print(f"Best composite improvement: {-best_score:.4f}")
        print(f"{'='*32}")
        
        return best_config, best_score, tuner
    @staticmethod
    def asha_search(args, dataset, current_run_dir, pretrainWE, 
                    config_space, initial_config,
                    num_trials=50, eta=2, r_min=50, r_max=200, prefact=1):
        """
        Run ASHA for ECRTM fine-tuning hyperparameters.
        
        Args:
            num_trials: Total number of HPO trials
            eta: Reduction factor (2 or 3 recommended)
            r_min: Minimum finetune epochs (e.g., 50)
            r_max: Maximum finetune epochs (e.g., 200)
            prefact: Multiplier for number of configs per rung
        """
        print(f"\n{'='*32}")
        print(f"Starting ASHA for ECRTM Fine-tuning")
        print(f"{'='*32}")
        print(f"Total trials: {num_trials}")
        print(f"eta={eta}, r_min={r_min}, r_max={r_max}, prefact={prefact}")
        print(f"Search space: {list(config_space.keys())}")
        print(f"Initial config: {initial_config}")
        
        # Create searcher and scheduler
        searcher = RandomSearcher(config_space, initial_config)
        scheduler = ASHAScheduler(
            searcher=searcher,
            eta=eta,
            r_min=r_min,
            r_max=r_max,
            prefact=prefact
        )
        
        print(f"Rung levels: {scheduler.rung_levels}")
        
        # Create objective function
        objective_fn = HPO.ecrtm_finetune_objective_fn(
            args, dataset, current_run_dir, pretrainWE
        )
        
        # Run HPO
        tuner = HPOTuner(scheduler=scheduler, objective_fn=objective_fn)
        tuner.run(number_of_trials=num_trials)
        
        # Get best config
        best_config, best_score = tuner.get_best_config()
        
        # Print rung statistics
        print(f"\n{'='*32}")
        print("ASHA Rung Statistics:")
        for rung in scheduler.rung_levels:
            n_completed = len(scheduler.completed_trials_at_rungs[rung])
            n_promoted = len(scheduler.promoted_configs[rung])
            print(f"  Rung {rung:3d}: {n_completed:3d} completed, {n_promoted:3d} promoted")
        
        print(f"\n{'='*32}")
        print("ASHA Results")
        print(f"{'='*32}")
        print(f"Best config:")
        for key, value in best_config.items():
            print(f"  {key}: {value:.4f}")
        print(f"Best composite improvement: {-best_score:.4f}")
        print(f"{'='*32}")
        
        return best_config, best_score, tuner

class HPOTuner:
    def __init__(self,scheduler,objective_fn:callable):
        self.scheduler = scheduler
        self.objective_fn = objective_fn
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.records = []
    def run(self,number_of_trials):
        for i in range(number_of_trials):
            print(f"\n{'='*32}")
            print(f"HPO Trial {i+1}/{number_of_trials}")
            print(f"{'='*32}")
            
            config = self.scheduler.suggest()
            print(f"Config: {config}")
            
            error = self.objective_fn(config)
            
            self.scheduler.update(config, error)
            self.bookkeeping(config, error)
            print(f"Validation error: {error:.4f}")
            print(f"Best error so far: {self.incumbent_error:.4f}")
    def bookkeeping(self,config,error):
        """Track best configuration and performance"""
        self.records.append({"config": config, "error": error})
        
        if self.incumbent is None or error < self.incumbent_error:
            self.incumbent = config
            self.incumbent_error = error
        
        self.incumbent_trajectory.append(self.incumbent_error)
    def get_best_config(self):
        return self.incumbent, self.incumbent_error
class RandomSearcher:
    def __init__(self,config_space:dict,initial_config:dict):
        self.config_space=config_space
        self.initial_config=initial_config
    def sample_config(self):
        if self.initial_config is not None:
            config=self.initial_config
            self.initial_config=None
            return config
        random_config={key:domain.rvs() for key,domain in self.config_space.items()}
        return random_config
    def update(self,config:dict,error:float,additional_info=None):
        pass
class BayesianSearcher:
    def __init__(self,config_space:dict,initial_config:dict,n_random_init:5):
        self.config_space=config_space
        self.initial_config=initial_config
        self.n_random_init=n_random_init
        self.trial_count=0
        self.X_observed=[] # Config sets
        self.y_observed=[] # Losses
        self.param_names=list(config_space.keys())
        self.bounds=[] # Bounds for each hyperparameter
        self.integer_params = []
        for key in self.param_names:
            domain=config_space[key]
            if 'epoch' in key:
                self.integer_params.append(key)
            self.bounds.append((domain.a,domain.a+domain.b)) # For domain, use scipy.stats
    def sample_config(self) -> dict:
        """Sample next configuration using BO or random initialization"""
        self.trial_count += 1
        
        # Use initial config first
        if self.initial_config is not None:
            config = self.initial_config
            self.initial_config = None
            return config
        
        # Random initialization phase
        if self.trial_count <= self.n_random_init:
            config={key: domain.rvs() for key, domain in self.config_space.items()}
            for key in self.integer_params:
                config[key]=int(round(config[key]))
            return config
        
        # Need at least 2 observations for GP
        if len(self.X_observed) < 2:
            config = {key: domain.rvs() for key, domain in self.config_space.items()}
            for key in self.integer_params:
                config[key]=int(round(config[key]))
            return config
        # Bayesian optimization phase
        try:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            
            # Fit Gaussian Process
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            gp.fit(X, y)
            
            # Optimize Expected Improvement
            next_x = self.optimize_acquisition(gp, y.min())
            
            # Convert array back to config dict
            config = {name: float(next_x[i]) for i, name in enumerate(self.param_names)}
            for key in self.integer_params:
                config[key]=int(round(config[key]))
            return config
        except Exception as e:
            print(f"  BO failed ({e}), falling back to random sampling")
            config = {key: domain.rvs() for key, domain in self.config_space.items()}
            for key in self.integer_params:
                config[key]=int(round(config[key]))
            return config 
    
    def update(self, config: dict, error: float, additional_info=None):
        """Record observation for GP fitting"""
        x = np.array([config[name] for name in self.param_names])
        self.X_observed.append(x)
        self.y_observed.append(error)
    
    def optimize_acquisition(self, gp, y_min, n_restarts=25):
        """Optimize Expected Improvement using L-BFGS-B"""
        best_x = None
        best_acquisition_value = -np.inf
        
        for _ in range(n_restarts):
            x0 = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            
            result = minimize(
                fun=lambda x: -self.expected_improvement(x, gp, y_min),
                x0=x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_acquisition_value:
                best_acquisition_value = -result.fun
                best_x = result.x
        
        return best_x if best_x is not None else x0
    
    def expected_improvement(self, x, gp, y_min, xi=0.01):
        """Expected Improvement acquisition function"""
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma == 0:
            return 0.0
        
        z = (y_min - mu - xi) / sigma
        ei = (y_min - mu - xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei
class BasicScheduler:
    def __init__(self,searcher):
        self.searcher=searcher
    def suggest(self):
        return self.searcher.sample_config()
    def update(self,config:dict,error:float,additional_info:None):
        self.searcher.update(config,error,additional_info)
class ASHAScheduler:
    def __init__(self,searcher,eta,r_min,r_max,prefact=1):
        self.searcher=searcher
        self.eta=eta
        self.r_min=r_min
        self.r_max=r_max
        self.prefact=prefact
        # Compute rung levels
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        self.rung_levels = [r_min * (eta**k) for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            self.rung_levels.append(r_max)
            self.K += 1
        
        # Track completed trials at each rung
        self.completed_trials_at_rungs = defaultdict(list)  # (config, error) pairs
        
        # Track which configs have been promoted from each rung
        self.promoted_configs = defaultdict(set)  # rung -> set of config hashes
        
        # Track number of configs started at each rung
        self.configs_started_at_rung = defaultdict(int)
    def _config_hash(self,config):
        items=[(k,v) for k,v in sorted(config.items()) if k!='finetune_epochs']
        return tuple(items)
    def suggest(self):
        """Suggest next configuration (either new or promoted)"""
        # Check rungs from top to bottom for promotion opportunities
        for i in range(len(self.rung_levels) - 2, -1, -1):
            rung = self.rung_levels[i]
            next_rung = self.rung_levels[i + 1]
            
            # Calculate how many configs should have been tried at this rung
            k = self.K - i
            n_required = int(self.prefact * (self.eta ** k))
            
            # Check if we have enough completed trials
            n_completed = len(self.completed_trials_at_rungs[rung])
            n_promoted = len(self.promoted_configs[rung])
            
            # Can we promote another config?
            if n_completed >= n_required and n_promoted < n_required:
                # Get best unpromoted config
                trials_at_rung = sorted(self.completed_trials_at_rungs[rung], key=lambda x: x[1])
                
                for config, error in trials_at_rung:
                    config_hash = self._config_hash(config)
                    if config_hash not in self.promoted_configs[rung]:
                        # Promote this config
                        self.promoted_configs[rung].add(config_hash)
                        promoted_config = config.copy()
                        promoted_config["finetune_epochs"] = next_rung
                        return promoted_config
        
        # No promotions available, sample new config at r_min
        config = self.searcher.sample_config()
        config["finetune_epochs"] = self.r_min
        return config
    def update(self, config: dict, error: float, info=None):
        """Update scheduler after trial completion"""
        ri = int(config["finetune_epochs"])
        
        # Update searcher
        self.searcher.update(config, error, additional_info=info)
        
        # Record completed trial
        self.completed_trials_at_rungs[ri].append((config, error))