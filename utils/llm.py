import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from utils.configs import Configs as cfg 


class LLM:
    def __init__(self, current_run_dir, num_top_words, vocab, sentence_embedder):
        load_dotenv()
        self.preference_dataset_path = os.path.join(current_run_dir, 'preference_dataset.jsonl')
        self.doc_topic_preference_dataset_path = os.path.join(current_run_dir, 'doc_topic_preference_dataset.jsonl')
        self.topic_descriptions_path = os.path.join(current_run_dir, 'topic_descriptions.txt')
        self.top_words_path = os.path.join(current_run_dir, f'top_words_{num_top_words}.jsonl')
        self.top_words_txt_path = os.path.join(current_run_dir, f'top_words_{num_top_words}.txt')
        self.theta_path = os.path.join(current_run_dir, 'train_theta.npy')
        self.vocab = vocab
        self.sentence_embedder = sentence_embedder
        self.document_embeddings = self.sentence_embedder.load_document_embeddings()
        self.model = cfg.LLM_MODEL
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.topic_word_system_prompt = cfg.TOPIC_WORD_SYSTEM_PROMPT
        self.topic_description_system_prompt = cfg.TOPIC_DESCRIPTION_SYSTEM_PROMPT

    def generate_topic_word_preference_dataset(self):
        def process_line(k, line):
            topic_data = json.loads(line.strip())
            top_words = topic_data['top_words']
            words_with_indices = []
            for word_dict in top_words:
                for word, idx in word_dict.items():
                    words_with_indices.append(f"'{word}' (beta_index: {idx})")
            
            prompt_content = f"Topic {k}: {', '.join(words_with_indices)}"
            
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.topic_word_system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.0
            )
            raw_data = response.choices[0].message.content.strip()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                raise TypeError(f"JSON parsing failed for line {k}: {raw_data}")
            return data
            
        with open(self.top_words_path, 'r', encoding='utf-8') as infile, open(self.preference_dataset_path, 'w', encoding='utf-8') as outfile:
            for k, line in enumerate(infile):
                data = process_line(k, line)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f'Created and saved preference dataset to: {self.preference_dataset_path}')

    def generate_doc_topic_preference_dataset(self):
        theta = np.load(self.theta_path)
        
        topic_description_embeddings = self.sentence_embedder.load_topic_description_embeddings()
        
        preference_dataset = []
        
        for doc_idx in range(theta.shape[0]):
            top_5_theta = np.argsort(theta[doc_idx])[-5:][::-1] # Descending order # @QUESTION: why need [::-1]?
            
            doc_embedding = self.document_embeddings[doc_idx].reshape(1, -1) # @QUESTION: why need reshape?
            similarities = cosine_similarity(doc_embedding, topic_description_embeddings)[0] # @QUESTION: why need [0]?
            top_5_semantic = np.argsort(similarities)[-5:][::-1] # Descending order
            
            set_theta = set(top_5_theta)
            set_semantic = set(top_5_semantic)
            
            t_plus_indices = list(set_theta.intersection(set_semantic)) # Good topics
            t_minus_indices = list(set_theta - set_semantic) # Bad topics
            
            # Only create preference if there are both good and bad topics
            if len(t_plus_indices) >= 1 and len(t_minus_indices) >= 1:
                preference_data = {
                    'd': int(doc_idx), 
                    't_plus_indices': [int(x) for x in t_plus_indices], 
                    't_minus_indices': [int(x) for x in t_minus_indices]   
                }
                preference_dataset.append(preference_data)
                
        with open(self.doc_topic_preference_dataset_path, 'w', encoding='utf-8') as f:
            for pref_data in preference_dataset:
                f.write(json.dumps(pref_data, ensure_ascii=False) + '\n')
        
        print(f'Created and saved doc-topic preference dataset to: {self.doc_topic_preference_dataset_path}')
        print(f'Generated {len(preference_dataset)} document-topic preference pairs')
    
    def generate_topic_descriptions(self):
        """Read self.top_words_txt_path and generate description for each topic
        """
        def process_topic_words_str(k, top_words_str):
            top_words = top_words_str.strip().split()
            
            prompt_content = f"Topic {k}: {', '.join(top_words)}"
            
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.topic_description_system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            raw_data = response.choices[0].message.content.strip()
            try: 
                data = json.loads(raw_data)
                topic_description = f"Topic name: {data['topic_name']}. Description: This topic is about {data['description']}. This topic includes words like: {data['key_words']}."
            except json.JSONDecodeError:
                raise TypeError(f"JSON parsing failed for line {k}: {raw_data}")
            return topic_description
                    
        with open(self.top_words_txt_path, 'r') as infile, \
             open(self.topic_descriptions_path, 'w', encoding='utf-8') as outfile:
                 
            for k, topic_words_str in enumerate(infile):
                topic_description = process_topic_words_str(k, topic_words_str)
                outfile.write(topic_description + '\n')
                    
        print(f'Created and saved topic descriptions to: {self.topic_descriptions_path}')
            