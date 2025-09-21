from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from itertools import combinations
from datasethandler.file_utils import split_text_word
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary,
                        topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return cv_per_topic, score

def compute_topic_diversity(top_words, _type='TD'):
    split_top_words = split_text_word(top_words)
    
    if _type == 'TD':
        # Topic Diversity
        unique_words = set()
        for topic in split_top_words:
            unique_words.update(topic)
        
        total_words = sum(len(topic) for topic in split_top_words)
        return len(unique_words) / total_words
    
    return 0.0

def TC_on_wikipedia(top_word_path, cv_type='C_V'):
    """
    Compute the TC score on the Wikipedia dataset
    """
    jar_dir = "evaluations"
    wiki_dir = os.path.join(".", 'datasets')
    random_number = np.random.randint(100000)
    
    os.system(
        f"java -jar {os.path.join(jar_dir, 'pametto.jar')} {os.path.join(wiki_dir, 'wikipedia/wikipedia_bd/')} {cv_type} {top_word_path} > tmp{random_number}.txt")
    cv_score = []
    with open(f"tmp{random_number}.txt", "r") as f:
        for line in f.readlines():
            if not line.startswith("202"):
                try:
                    cv_score.append(float(line.strip().split()[1]))
                except:
                    continue
    os.remove(f"tmp{random_number}.txt")
    return cv_score, sum(cv_score) / len(cv_score)

def evaluate_clustering(test_theta, test_labels):
    # K-means clustering
    n_clusters = len(np.unique(test_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(test_theta)
    
    # Calculate metrics
    nmi = normalized_mutual_info_score(test_labels, predicted_labels)
    
    # Calculate purity
    def purity_score(y_true, y_pred):
        contingency_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(len(y_true)):
            contingency_matrix[y_true[i]][y_pred[i]] += 1
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
    purity = purity_score(test_labels, predicted_labels)
    
    return {'NMI': nmi, 'Purity': purity}

def evaluate_classification(train_theta, test_theta, train_labels, test_labels, tune=False):
    if tune:
        # Grid search for best parameters
        param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
        svm = GridSearchCV(SVC(random_state=42), param_grid, cv=5)
    else:
        svm = SVC(random_state=42)
    
    # Train classifier
    svm.fit(train_theta, train_labels)
    
    # Predict
    predicted_labels = svm.predict(test_theta)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    macro_f1 = f1_score(test_labels, predicted_labels, average='macro')
    
    return {'acc': accuracy, 'macro-F1': macro_f1}
#!/usr/bin/env python3
"""
Debug script to evaluate finetuned topic model results
"""

import os
import sys
import numpy as np
import wandb
from dotenv import load_dotenv

# Set correct path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import datasethandler.basic_dataset_handler as datasethandler

def load_top_words_from_file(file_path):
    """Load top words from text file"""
    with open(file_path, 'r') as f:
        top_words = f.read().strip()
    return top_words

def main():
    load_dotenv()
    
    # Configuration
    RESULT_DIR = 'results/ECRTM/StackOverflow/2025-09-21_18-32-34'
    DATA_DIR = 'datasets/StackOverflow'
    
    # Setup wandb
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project='finetuned-evaluation', config={'result_dir': RESULT_DIR})
    
    print("Loading dataset...")
    # Load dataset
    dataset = datasethandler.BasicDatasetHandler(
        DATA_DIR, device='cpu', read_labels=True,
        as_tensor=True, contextual_embed=False)
    
    print("Loading finetuned top words...")
    # Load finetuned top words
    top_words_10 = load_top_words_from_file(os.path.join(RESULT_DIR, 'finetuned_top_words_10.txt'))
    top_words_15 = load_top_words_from_file(os.path.join(RESULT_DIR, 'finetuned_top_words_15.txt'))
    top_words_20 = load_top_words_from_file(os.path.join(RESULT_DIR, 'finetuned_top_words_20.txt'))
    top_words_25 = load_top_words_from_file(os.path.join(RESULT_DIR, 'finetuned_top_words_25.txt'))
    
    print("Loading theta matrices...")
    # Load theta matrices
    train_theta = np.load(os.path.join(RESULT_DIR, 'train_theta.npy'))
    test_theta = np.load(os.path.join(RESULT_DIR, 'test_theta.npy'))

    print("Computing theta argmax statistics...")
    # Theta argmax analysis
    train_theta_argmax = train_theta.argmax(axis=1)
    unique_elements, counts = np.unique(train_theta_argmax, return_counts=True)
    print(f'train theta argmax: {unique_elements, counts}')
    
    test_theta_argmax = test_theta.argmax(axis=1)
    unique_elements, counts = np.unique(test_theta_argmax, return_counts=True)
    print(f'test theta argmax: {unique_elements, counts}')

    print("Computing Topic Diversity metrics...")
    # Topic Diversity
    TD_10 = compute_topic_diversity(top_words_10, _type="TD")
    print(f"TD_10: {TD_10:.5f}")
    wandb.log({"TD_10": TD_10})

    TD_15 = compute_topic_diversity(top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    wandb.log({"TD_15": TD_15})

    TD_20 = compute_topic_diversity(top_words_20, _type="TD")
    print(f"TD_20: {TD_20:.5f}")
    wandb.log({"TD_20": TD_20})

    TD_25 = compute_topic_diversity(top_words_25, _type="TD")
    print(f"TD_25: {TD_25:.5f}")
    wandb.log({"TD_25": TD_25})

    print("Computing clustering metrics...")
    # Clustering evaluation
    clustering_results = evaluate_clustering(test_theta, dataset.test_labels)
    print(f"NMI: {clustering_results['NMI']}")
    print(f"Purity: {clustering_results['Purity']}")
    wandb.log({"NMI": clustering_results['NMI']})
    wandb.log({"Purity": clustering_results['Purity']})

    print("Computing classification metrics...")
    # Classification evaluation
    classification_results = evaluate_classification(
        train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=False)
    print(f"Accuracy: {classification_results['acc']}")
    print(f"Macro-f1: {classification_results['macro-F1']}")
    wandb.log({"Accuracy": classification_results['acc']})
    wandb.log({"Macro-f1": classification_results['macro-F1']})

    print("Computing Topic Coherence (Wikipedia)...")
    # Topic Coherence on Wikipedia
    try:
        TC_15_list, TC_15 = TC_on_wikipedia(
            os.path.join(RESULT_DIR, 'finetuned_top_words_15.txt'))
        print(f"TC_15: {TC_15:.5f}")
        wandb.log({"TC_15": TC_15})
    except Exception as e:
        print(f"TC_15 computation failed: {e}")

    try:
        TC_10_list, TC_10 = TC_on_wikipedia(
            os.path.join(RESULT_DIR, 'finetuned_top_words_10.txt'))
        print(f"TC_10: {TC_10:.5f}")
        wandb.log({"TC_10": TC_10})
    except Exception as e:
        print(f"TC_10 computation failed: {e}")

    print("Computing NPMI metrics...")
    # NPMI on training corpus
    try:
        NPMI_train_10_list, NPMI_train_10 = compute_topic_coherence(
            dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_npmi')
        print(f"NPMI_train_10: {NPMI_train_10:.5f}")
        wandb.log({"NPMI_train_10": NPMI_train_10})
    except Exception as e:
        print(f"NPMI_train_10 computation failed: {e}")

    try:
        NPMI_train_15_list, NPMI_train_15 = compute_topic_coherence(
            dataset.train_texts, dataset.vocab, top_words_15, cv_type='c_npmi')
        print(f"NPMI_train_15: {NPMI_train_15:.5f}")
        wandb.log({"NPMI_train_15": NPMI_train_15})
    except Exception as e:
        print(f"NPMI_train_15 computation failed: {e}")

    # NPMI on Wikipedia
    try:
        NPMI_wiki_10_list, NPMI_wiki_10 = TC_on_wikipedia(
            os.path.join(RESULT_DIR, 'finetuned_top_words_10.txt'), cv_type='NPMI')
        print(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}")
        wandb.log({"NPMI_wiki_10": NPMI_wiki_10})
    except Exception as e:
        print(f"NPMI_wiki_10 computation failed: {e}")

    try:
        NPMI_wiki_15_list, NPMI_wiki_15 = TC_on_wikipedia(
            os.path.join(RESULT_DIR, 'finetuned_top_words_15.txt'), cv_type='NPMI')
        print(f"NPMI_wiki_15: {NPMI_wiki_15:.5f}")
        wandb.log({"NPMI_wiki_15": NPMI_wiki_15})
    except Exception as e:
        print(f"NPMI_wiki_15 computation failed: {e}")

    # C_P coherence on Wikipedia
    try:
        Cp_wiki_10_list, Cp_wiki_10 = TC_on_wikipedia(
            os.path.join(RESULT_DIR, 'finetuned_top_words_10.txt'), cv_type='C_P')
        print(f"Cp_wiki_10: {Cp_wiki_10:.5f}")
        wandb.log({"Cp_wiki_10": Cp_wiki_10})
    except Exception as e:
        print(f"Cp_wiki_10 computation failed: {e}")

    try:
        Cp_wiki_15_list, Cp_wiki_15 = TC_on_wikipedia(
            os.path.join(RESULT_DIR, 'finetuned_top_words_15.txt'), cv_type='C_P')
        print(f"Cp_wiki_15: {Cp_wiki_15:.5f}")
        wandb.log({"Cp_wiki_15": Cp_wiki_15})
    except Exception as e:
        print(f"Cp_wiki_15 computation failed: {e}")

    print("Evaluation completed! Check wandb dashboard for results.")
    wandb.finish()

if __name__ == "__main__":
    main()