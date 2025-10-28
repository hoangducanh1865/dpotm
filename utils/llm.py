import logging
import re
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from utils.config import Config 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LLM:
    def __init__(self, current_run_dir, num_top_words, vocab, args, doc_topic_preference_dataset_generation_logger):
        load_dotenv()
        self.args=args
        self.topic_word_preference_dataset_path = os.path.join(current_run_dir, 'topic_word_preference_dataset.jsonl')
        self.doc_topic_preference_dataset_path = os.path.join('data', 'doc_topic_preference_dataset', 'doc_topic_preference_dataset.jsonl')
        self.topic_descriptions_path = os.path.join(current_run_dir, 'topic_descriptions.txt')
        self.top_words_path = os.path.join(current_run_dir, f'top_words_{num_top_words}.jsonl')
        self.top_words_txt_path = os.path.join(current_run_dir, f'top_words_{num_top_words}.txt')
        self.theta_path = os.path.join(current_run_dir, 'train_theta.npy')
        self.train_text_path = os.path.join('datasets', args.dataset, 'train_texts.txt')
        self.vocab = vocab
        self.model = Config.LLM_MODEL
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.topic_word_system_prompt = Config.TOPIC_WORD_SYSTEM_PROMPT
        self.topic_description_system_prompt = Config.TOPIC_DESCRIPTION_SYSTEM_PROMPT
        self.hf_model_name = Config.HF_MODEL
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_token = os.getenv('HF_TOKEN')
        self.logger = logging.getLogger('main')
        self.log_llm_predictions_first_time = True
        self.doc_topic_preference_dataset_generation_logger=doc_topic_preference_dataset_generation_logger

    def generate_topic_word_preference_dataset(self):
        if self.args.topic_word_dataset_generate_method == 'openai':
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
                ).choices[0].message.content.strip() # Just take raw content
                
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    raise TypeError(f"JSON parsing failed for line {k}: {response}")
                return data
                
            with open(self.top_words_path, 'r', encoding='utf-8') as infile, open(self.topic_word_preference_dataset_path, 'w', encoding='utf-8') as outfile:
                for k, line in tqdm(enumerate(infile),total=self.args.num_topics):
                    data = process_line(k, line)
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        elif self.args.topic_word_dataset_generate_method == 'hf':
            # @TODO
            pass
    
        else:
            raise NotImplementedError('Generate topic word preference dataset method not supported')
        
        print(f'Created and saved topic-word preference dataset to: {self.topic_word_preference_dataset_path}')
        
    def print_llm_predictions(self):
        self.logger.info('LLM predictions on train set')
        predictions = []
        with open(self.doc_topic_preference_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    predictions.append(str(data['ranking'][0]))
        
        pred_str = ' '.join(predictions)
        self.logger.info(pred_str)

    def generate_doc_topic_preference_dataset(self):
        theta = np.load(self.theta_path)
        
        # Check if the dataset are already exist, then ignore this function call time
        if os.path.exists(self.doc_topic_preference_dataset_path):
            with open(self.doc_topic_preference_dataset_path, 'r') as f:
                processed_lines = [line.strip() for line in f if line.strip()]
            
            if len(processed_lines) >= theta.shape[0]:
                print(f'Doc-topic preference dataset already complete')
                if self.log_llm_predictions_first_time:
                    self.print_llm_predictions()
                    self.log_llm_predictions_first_time = False
                return
            else:
                print(f'Doc-topic preference dataset has not complete yet ({len(processed_lines)}/{theta.shape[0]}), continuing the process...')
                batch_continue = json.loads(processed_lines[-1])['doc_index'] + 1
        else:
            batch_continue = 0
            
            # Create/clear the file        
            os.makedirs(os.path.dirname(self.doc_topic_preference_dataset_path), exist_ok=True)
            with open(self.doc_topic_preference_dataset_path, 'w', encoding='utf-8') as f:
                pass  
        
        if self.args.doc_topic_dataset_generate_method == 'hf':
            self.__setup_hf_llm()
        elif self.args.doc_topic_dataset_generate_method == 'openai':
            # @TODO: setup OpenAI chat here?
            pass
        
        with open(self.train_text_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        
        with open(self.topic_descriptions_path, 'r', encoding='utf-8') as f:
            topic_descriptions = [line.strip() for line in f if line.strip()]
        
        topic_descriptions_str = '\n'.join([
            f'{i}: {description}' for i, description in enumerate(topic_descriptions)
        ])
                    
        doc_topic_preference_dataset = []
        failed_doc_count = 0
        
        with tqdm(total=theta.shape[0] - batch_continue) as pbar:
            for batch_start in range(batch_continue, theta.shape[0], self.args.num_docs_per_call):
                batch_end = min(batch_start + self.args.num_docs_per_call, theta.shape[0])
                batch_docs = []
                
                '''print('')
                print('=' * 32)
                print(f'Process doc {batch_start} to doc {batch_end - 1}')'''
                
                for doc_idx in range(batch_start, batch_end):
                    doc_text = documents[doc_idx]
                    top_5_topic_indices = np.argsort(theta[doc_idx])[-5:][::-1]
                    batch_docs.append({'doc_idx': doc_idx, 'doc_text': doc_text, 'top_5_topic_indices': top_5_topic_indices})

                prompt = Config.get_doc_topic_prompt(topic_descriptions_str, batch_docs)
                                
                try:
                    
                    if self.args.doc_topic_dataset_generate_method == 'hf':
                        '''response = self.hf_model(prompt, max_new_tokens=32768, temperature=0.7, top_p=0.9)[0]["generated_text"]'''
                        messages = [
                            {"role": "user", "content": prompt}
                        ]
                        text = self.hf_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                        )
                        model_inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.hf_model.device)

                        # conduct text completion
                        generated_ids = self.hf_model.generate(
                            **model_inputs,
                            max_new_tokens=32768
                        )
                        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

                        # parsing thinking content
                        try:
                            # rindex finding 151668 (</think>)
                            index = len(output_ids) - output_ids[::-1].index(151668)
                        except ValueError:
                            index = 0

                        '''thinking_content = self.hf_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")'''
                        response = self.hf_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                        
                    elif self.args.doc_topic_dataset_generate_method == 'openai':                
                        response = self.llm.chat.completions.create(
                            model=self.model,
                            messages=[
                                {'role': 'system', 'content': 'You are an expert at ranking topics by relevance to documents. Return rankings as specified in the format.'},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=0.3,
                            max_tokens=8192
                        ).choices[0].message.content.strip() # Just take raw content
                    
                    else:
                        print('Generate doc topic preference dataset method not supported, fallback to OpenAI API method')
                        
                        response = self.llm.chat.completions.create(
                            model=self.model,
                            messages=[
                                {'role': 'system', 'content': 'You are an expert at ranking topics by relevance to documents. Return rankings as specified in the format.'},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=0.3,
                            max_tokens=8192
                        ).choices[0].message.content.strip() # Just take raw content
                    
                    '''print(f"Raw response: {repr(response)}")''' # Debug

                    response_lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
                    
                    for i, doc_data in enumerate(batch_docs):
                        doc_idx = doc_data['doc_idx']
                        top_5_topic_indices = doc_data['top_5_topic_indices']
                        
                        line = response_lines[i]
                        numbers = re.findall(r'\b\d+\b', line)
                        
                        ranking = []
                        if numbers:
                            for num_str in numbers:
                                num = int(num_str)
                                if num in top_5_topic_indices:
                                    ranking.append(num)
                            
                            if len(ranking) != 5:
                                self.doc_topic_preference_dataset_generation_logger.info('⚠️ Different from 5 numbers found in this line, use theta order instead')
                                ranking = top_5_topic_indices.tolist()
                                failed_doc_count += 1
                            else:
                                self.doc_topic_preference_dataset_generation_logger.info(f'Doc {doc_idx} extracted ranking: {ranking}')
                        else:
                            self.doc_topic_preference_dataset_generation_logger.info(f'⚠️ No numbers found for doc {doc_idx}, use theta order instead')
                            ranking = top_5_topic_indices.tolist()
                            failed_doc_count += 1
                        
                        doc_topic_preference_dataset.append({
                            "doc_index": doc_idx,
                            "top_5_topics": top_5_topic_indices.tolist(),
                            "ranking": ranking
                        })
                            
                except Exception as e:
                    self.doc_topic_preference_dataset_generation_logger.info(f"⚠️ Batch failed: {e}")
                    # Just add un-processed document
                    for i, doc_data in enumerate(batch_docs):
                        doc_idx = doc_data['doc_idx']
                        
                        # Check if this document was processed or not
                        if not any(item['doc_index'] == doc_idx for item in doc_topic_preference_dataset):
                            top_5_topic_indices = doc_data['top_5_topic_indices']
                            doc_topic_preference_dataset.append({
                                "doc_index": doc_idx,
                                "top_5_topics": top_5_topic_indices.tolist(),
                                "ranking": top_5_topic_indices.tolist()
                            })
                            failed_doc_count += 1
                
                with open(self.doc_topic_preference_dataset_path, 'a', encoding='utf-8') as f:
                    for data in doc_topic_preference_dataset:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                doc_topic_preference_dataset = []  
                pbar.update(batch_end-batch_start)

        print(f'Created and saved doc-topic preference dataset to: {self.doc_topic_preference_dataset_path}')
        print(f'Successful document processing rate: {((len(documents) - failed_doc_count) / len(documents)) * 100:.2f}%')
        if self.log_llm_predictions_first_time:
            self.print_llm_predictions()
            self.log_llm_predictions_first_time = False
            
    def generate_topic_descriptions(self):
        """Read self.top_words_txt_path and generate description for each topic
        """
        if self.args.generate_topic_descriptions_method == 'openai':
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
                    
                for k, topic_words_str in tqdm(enumerate(infile),total=self.args.num_topics):
                    topic_description = process_topic_words_str(k, topic_words_str)
                    outfile.write(topic_description + '\n')
        
        elif self.args.generate_topic_descriptions_method == 'hf':
            # @TODO
            pass
        
        else:
            raise NotImplementedError('Generate topic descriptions method not supported')
                    
        print(f'Created and saved topic descriptions to: {self.topic_descriptions_path}')

    def __setup_hf_llm(self):
        print(f"Loading Hugging Face model: {self.hf_model_name}")
        
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        print("Load Hugging Face model successfully!")