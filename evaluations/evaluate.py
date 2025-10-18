import os
import numpy as np
import evaluations
import wandb
def evaluate(trainer, train_theta, test_theta, logger, read_labels, dataset, args, current_run_dir, suffix=''):
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir, suffix)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir, suffix)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir, suffix)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir, suffix)

    # argmax of train and test theta
    # train_theta_argmax = train_theta.argmax(axis=1)
    # test_theta_argmax = test_theta.argmax(axis=1) 
    train_theta_argmax = train_theta.argmax(axis=1)
    unique_elements, counts = np.unique(train_theta_argmax, return_counts=True)
    print(f'train theta argmax: {unique_elements, counts}')
    logger.info(f'train theta argmax: {unique_elements, counts}')
    test_theta_argmax = test_theta.argmax(axis=1)
    unique_elements, counts = np.unique(test_theta_argmax, return_counts=True)
    print(f'test theta argmax: {unique_elements, counts}')
    logger.info(f'test theta argmax: {unique_elements, counts}')       

    # TD_15 = evaluations.compute_topic_diversity(
    #     top_words_15, _type="TD")
    # print(f"TD_15: {TD_15:.5f}")


    # # evaluating clustering
    # if read_labels:
    #     clustering_results = evaluations.evaluate_clustering(
    #         test_theta, dataset.test_labels)
    #     print(f"NMI: ", clustering_results['NMI'])
    #     print(f'Purity: ', clustering_results['Purity'])


    # TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
    #     os.path.join(current_run_dir, 'top_words_15.txt'))
    # print(f"TC_15: {TC_15:.5f}")
    TD_10 = evaluations.compute_topic_diversity(
        top_words_10, _type="TD")
    print(f"TD_10: {TD_10:.5f}")
    wandb.log({"TD_10": TD_10})
    logger.info(f"TD_10: {TD_10:.5f}")

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    wandb.log({"TD_15": TD_15})
    logger.info(f"TD_15: {TD_15:.5f}")

    # TD_20 = topmost.evaluations.compute_topic_diversity(
    #     top_words_20, _type="TD")
    # print(f"TD_20: {TD_20:.5f}")
    # wandb.log({"TD_20": TD_20})
    # logger.info(f"TD_20: {TD_20:.5f}")

    # TD_25 = topmost.evaluations.compute_topic_diversity(
    #     top_words_25, _type="TD")
    # print(f"TD_25: {TD_25:.5f}")
    # wandb.log({"TD_25": TD_25})
    # logger.info(f"TD_25: {TD_25:.5f}")
    
    # Calculate IRBO
    def process_top_words(top_words_list) -> list[list[str]]:
        res = []
        for top_words in top_words_list:
            top_words = top_words.split() # string of words to list of words
            res.append(top_words)
        return res 
    
    IBRO_10 = evaluations.buubyyboo_dth(top_words=process_top_words(top_words_10), topk=10, weight=0.9)
    IBRO_15 = evaluations.buubyyboo_dth(top_words=process_top_words(top_words_15), topk=15, weight=0.9)
    print(f'IBRO_10: {IBRO_10:.4f}')
    print(f'IBRO_15: {IBRO_15:.4f}')

    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        wandb.log({"NMI": clustering_results['NMI']})
        wandb.log({"Purity": clustering_results['Purity']})
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")

    # evaluate classification
    '''if read_labels:
        classification_results = evaluations.evaluate_classification(
            train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=args.tune_SVM)
        print(f"Accuracy: ", classification_results['acc'])
        wandb.log({"Accuracy": classification_results['acc']})
        logger.info(f"Accuracy: {classification_results['acc']}")
        print(f"Macro-f1", classification_results['macro-F1'])
        wandb.log({"Macro-f1": classification_results['macro-F1']})
        logger.info(f"Macro-f1: {classification_results['macro-F1']}")'''

    # TC
    '''TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        use_kaggle=args.use_kaggle,
        use_colab=args.use_colab,
        top_word_path=os.path.join(current_run_dir, suffix + 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")
    wandb.log({"TC_15": TC_15})
    logger.info(f"TC_15: {TC_15:.5f}")
    logger.info(f'TC_15 list: {TC_15_list}')'''

    # TC_10_list, TC_10 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
    #     os.path.join(current_run_dir, 'top_words_10.txt'))
    # print(f"TC_10: {TC_10:.5f}")
    # wandb.log({"TC_10": TC_10})
    # logger.info(f"TC_10: {TC_10:.5f}")
    # logger.info(f'TC_10 list: {TC_10_list}')

    # NPMI
    '''NPMI_train_10_list, NPMI_train_10 = evaluations.compute_topic_coherence(
        dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_npmi')
    print(f"NPMI_train_10: {NPMI_train_10:.5f}, NPMI_train_10_list: {NPMI_train_10_list}")
    wandb.log({"NPMI_train_10": NPMI_train_10})
    logger.info(f"NPMI_train_10: {NPMI_train_10:.5f}")
    logger.info(f'NPMI_train_10 list: {NPMI_train_10_list}')

    NPMI_wiki_10_list, NPMI_wiki_10 = evaluations.topic_coherence.TC_on_wikipedia(
        use_kaggle=args.use_kaggle, 
        use_colab=args.use_colab,
        top_word_path=os.path.join(current_run_dir, f'{suffix}top_words_10.txt'), 
        cv_type='NPMI')
    print(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}, NPMI_wiki_10_list: {NPMI_wiki_10_list}")
    wandb.log({"NPMI_wiki_10": NPMI_wiki_10})
    logger.info(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}")
    logger.info(f'NPMI_wiki_10 list: {NPMI_wiki_10_list}')

    Cp_wiki_10_list, Cp_wiki_10 = evaluations.topic_coherence.TC_on_wikipedia(
        use_kaggle=args.use_kaggle, 
        use_colab=args.use_colab,
        top_word_path=os.path.join(current_run_dir, f'{suffix}top_words_10.txt'), 
        cv_type='C_P')
    print(f"Cp_wiki_10: {Cp_wiki_10:.5f}, Cp_wiki_10_list: {Cp_wiki_10_list}")
    wandb.log({"Cp_wiki_10": Cp_wiki_10})
    logger.info(f"Cp_wiki_10: {Cp_wiki_10:.5f}")
    logger.info(f'Cp_wiki_10 list: {Cp_wiki_10_list}')'''
    
    # w2v_list, w2v = evaluations.topic_coherence.compute_topic_coherence(
    #     dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_w2v')
    # print(f"w2v: {w2v:.5f}, w2v_list: {w2v_list}")
    # wandb.log({"w2v": w2v})
    # logger.info(f"w2v: {w2v:.5f}")
    # logger.info(f'w2v list: {w2v_list}')