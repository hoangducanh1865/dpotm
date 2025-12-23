import sys
import os
import numpy as np
import basic_trainer
import datasethandler
import scipy
import wandb
from dotenv import load_dotenv
from models.ECRTM.ECRTM import ECRTM
from utils.llm import LLM
from utils import config, log, miscellaneous, seed
from utils.config import Config 
from evaluations.evaluate import evaluate
from utils.hpo import HPO
from scipy import stats

RESULT_DIR = 'results'
DATA_DIR = 'datasets'


if __name__ == "__main__":
    load_dotenv()
    
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_logging_argument(parser)
    config.add_model_argument(parser)
    config.add_wete_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    args = parser.parse_args()
    
    prj = args.wandb_prj if args.wandb_prj else 'baselines'

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR + "/" + str(args.model) + "/" +str(args.dataset), current_time)
    current_checkpoint_dir = os.path.join(current_run_dir, "checkpoints")
    os.makedirs(current_checkpoint_dir, exist_ok=True) # QUESTION: is this necessary?
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)
    
    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    doc_topic_preference_dataset_generation_logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'doc_topic_preference_dataset_generation.log'))
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project=prj, config=args)
    wandb.log({'time_stamp': current_time})

    # if args.dataset in ['YahooAnswers']:
    #     read_labels = True
    # else:
    #     read_labels = False
    read_labels = True

    # load a preprocessed dataset
    dataset = datasethandler.BasicDatasetHandler(
        args=args,
        dataset_dir=os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=True)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()


    model = ECRTM(args,
                    vocab = dataset.vocab,
                    vocab_size=dataset.vocab_size, 
                    num_topics=args.num_topics, 
                    dropout=args.dropout, 
                    pretrained_WE=pretrainWE if args.use_pretrainWE else None, 
                    weight_loss_ECR=args.weight_ECR,
                    current_run_dir=current_run_dir)

    # create a trainer
    trainer = basic_trainer.BasicTrainer(model, epochs=args.epochs,
                                        learning_rate=args.lr,
                                        batch_size=args.batch_size,
                                        use_lr_scheduler=args.lr_scheduler,
                                        lr_step_size=args.lr_step_size,
                                        device=args.device,
                                        args=args,
                                        checkpoint_dir=current_checkpoint_dir,
                                        dataset=dataset,
                                        current_run_dir=current_run_dir)
    def evaluate_during_training(epoch):
        tmp_train_theta = trainer.test(dataset.train_data)
        tmp_test_theta = trainer.test(dataset.test_data)
        evaluate(trainer, tmp_train_theta, tmp_test_theta, logger, read_labels, dataset, args, current_run_dir, suffix=f'epoch_{epoch}_')
        
    # train model
    if args.checkpoint_path:
        print('Loading checkpoint...')
        start_epoch = trainer.load_checkpoint(args.checkpoint_path) 
    else:
        print('Training model...')
        trainer.train(dataset, 1, args.epochs, evaluate_fn=evaluate_during_training)
        
    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    
    evaluate(trainer, train_theta, test_theta, logger, read_labels, dataset, args, current_run_dir)

    if args.finetune_beta == 0 and args.finetune_theta == 0:
        wandb.finish()
        sys.exit(0)
        
    # Train as usual and then find best configuration
    if args.run_hpo:
        if args.hpo_algorithm=='asha':
            config_space={
                'weight_doc_topic_dpo':stats.uniform(0.0,50.0),
                'weight_doc_topic_reg':stats.uniform(0.0,50.0),
                # 'finetune_epochs':stats.randint(50,300) # finetune_epochs is already controlled by ASHA scheduler
            }
            initial_config={
                'weight_doc_topic_dpo':args.weight_doc_topic_dpo,
                'weight_doc_topic_reg':args.weight_doc_topic_reg,
                'finetune_epochs':args.finetune_epochs
            }
            best_config,best_score,tuner=HPO.asha_search(
                args, dataset, current_run_dir, pretrainWE, 
                config_space, initial_config,
                num_trials=50, eta=2, r_min=50, r_max=200, prefact=1
            )
        elif args.hpo_algorithm=='bayesian':
            config_space={
                'weight_doc_topic_dpo':stats.uniform(0.0,50.0),
                'weight_doc_topic_reg':stats.uniform(0.0,50.0),
                'finetune_epochs':stats.uniform(50,300) 
            }
            initial_config={
                'weight_doc_topic_dpo':args.weight_doc_topic_dpo,
                'weight_doc_topic_reg':args.weight_doc_topic_reg,
                'finetune_epochs':float(args.finetune_epochs)
            }
            best_config,best_score,tuner=HPO.bayesian_search(
                args, dataset, current_run_dir, pretrainWE,
                config_space=config_space,
                initial_config=initial_config,
                num_trials=20,
                n_random_init=5
            )
            best_config['finetune_epochs']=int(round(best_config['finetune_epochs']))
        else:
            raise NotImplementedError('HPO algorithm not supported')
        args.weight_doc_topic_dpo=best_config['weight_doc_topic_dpo']
        args.weight_doc_topic_reg=best_config['weight_doc_topic_reg']
        args.finetune_epochs=best_config['finetune_epochs']
        print(f'Best weight_doc_topic_dpo: {best_config["weight_doc_topic_dpo"]}')
        print(f'Best weight_doc_topic_reg: {best_config["weight_doc_topic_reg"]}')
        print(f'Best finetune_epochs: {best_config["finetune_epochs"]}')
        
    # LLM and Sentence Transformer models
    trainer.model.is_finetuning = True
    trainer.llm = LLM(current_run_dir, args.num_top_words, dataset.vocab, args,doc_topic_preference_dataset_generation_logger)
    if args.finetune_beta:
        print('Generating topic-word preference dataset...')
        trainer.llm.generate_topic_word_preference_dataset()
        trainer.model.load_topic_word_preference_dataset()
    if args.finetune_theta:
        print('Generating topic descriptions...')
        trainer.llm.generate_topic_descriptions()
        print(f'Generating doc-topic preference dataset, with batch_size is {args.num_docs_per_call}...')
        trainer.llm.generate_doc_topic_preference_dataset()  
        trainer.model.load_doc_topic_preference_dataset()
    # Fine-tune model
    if args.checkpoint_path:
        print('Fine-tuning model...')
        trainer.train(dataset, start_epoch, start_epoch - 1 + args.finetune_epochs, evaluate_fn=evaluate_during_training) 
    else:
        print('Fine-tuning model...')
        trainer.train(dataset, args.epochs + 1, args.epochs + args.finetune_epochs, evaluate_fn=evaluate_during_training) 
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    evaluate(trainer, train_theta, test_theta, logger, read_labels, dataset, args, current_run_dir, suffix='finetuned_')
    
    wandb.finish()
