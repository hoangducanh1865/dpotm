import gc
import torch
import os
import json
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, BitsAndBytesConfig
from utils.config import Config as cfg 


class LLM:
    def __init__(self, current_run_dir, num_top_words, vocab, sentence_embedder, model_type='openai_api'):
        load_dotenv()
        self.preference_dataset_path = os.path.join(current_run_dir, 'preference_dataset.jsonl')
        self.doc_topic_preference_dataset_path = os.path.join(current_run_dir, 'doc_topic_preference_dataset.jsonl')
        self.topic_descriptions_path = os.path.join(current_run_dir, 'topic_descriptions.txt')
        self.top_words_path = os.path.join(current_run_dir, f'top_words_{num_top_words}.jsonl')
        self.top_words_txt_path = os.path.join(current_run_dir, f'top_words_{num_top_words}.txt')
        self.theta_path = os.path.join(current_run_dir, 'train_theta.npy')
        self.vocab = vocab
        self.sentence_embedder = sentence_embedder
        self.document_embeddings = None
        self.documents = None
        
        # Model type selection
        self.model_type = model_type # ['openai_api', 'hf_model']
        
        # Local Hugging Face LM setup
        self.hf_model_name = cfg.HF_MODEL
        self.hf_token = os.getenv('HF_TOKEN')
        self.hf_tokenizer = None
        self.hf_model = None 
        self.hf_pipeline = None
        
        # OpenAI setup
        self.openai_model_name = cfg.OPENAI_MODEL
        self.openai_client = None
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # System prompts
        self.topic_word_system_prompt = cfg.TOPIC_WORD_SYSTEM_PROMPT
        self.topic_description_system_prompt = cfg.TOPIC_DESCRIPTION_SYSTEM_PROMPT
        
        self.__setup_llm()

    def generate_topic_word_preference_dataset(self):
        def process_line(k, line):
            topic_data = json.loads(line.strip())
            top_words = topic_data['top_words']
            words_with_indices = []
            for word_dict in top_words:
                for word, idx in word_dict.items():
                    words_with_indices.append(f"'{word}' (beta_index: {idx})")
            
            prompt_content = f"Topic {k}: {', '.join(words_with_indices)}"
            
            raw_data = self.__generate_response(
                system_prompt=self.topic_word_system_prompt,
                user_prompt=prompt_content,
                max_new_tokens=200,
                temperature=0.3
            )
            
            if raw_data is None:
                raise RuntimeError(f'Fail to generate response for topic {k}')
            
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

    def generate_doc_topic_preference_dataset(self, method='similarity_search', num_top_topics=10):
        theta = np.load(self.theta_path)
        preference_dataset = []
            
        if method == 'similarity_search':
            if self.document_embeddings is None:
                self.document_embeddings = self.sentence_embedder.load_document_embeddings()
            topic_description_embeddings = self.sentence_embedder.load_topic_description_embeddings()
            
            for doc_idx in range(theta.shape[0]):
                top_theta = np.argsort(theta[doc_idx])[-num_top_topics:][::-1] # Descending order # @QUESTION: why need [::-1]?
                
                doc_embedding = self.document_embeddings[doc_idx].reshape(1, -1) # @QUESTION: why need reshape?
                similarities = cosine_similarity(doc_embedding, topic_description_embeddings)[0] # @QUESTION: why need [0]?
                top_semantic = np.argsort(similarities)[-num_top_topics:][::-1] # Descending order
                
                set_theta = set(top_theta)
                set_semantic = set(top_semantic)
                
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

        elif method == 'llm':            
            if self.documents is None:
                self.documents = self.sentence_embedder.load_documents()
            topic_descriptions = self.sentence_embedder.load_topic_descriptions()
            
            for doc_idx in range(len(self.documents)):
                # Top 10 topics from theta
                top_theta_indices = np.argsort(theta[doc_idx])[-num_top_topics:][::-1]
                
                doc_text = self.documents[doc_idx]
                
                # Get relevance scores of topics from LLM
                topic_relevance_scores = self.__get_topic_relevance_scores(doc_text, top_theta_indices, topic_descriptions) # -> List[(topic_idx, relevance_score)]
                
                topic_relevance_scores.sort(key=lambda x: x[1], reverse=True) # @QUESTION: Why do complicated like this? Why not just simply sort()?
                
                # Create preference dataset: top 30% is good, bottom 30% is bad
                cutoff = max(1, num_top_topics // 3)
                
                t_plus_indices = [topic_idx for topic_idx, _ in topic_relevance_scores[:cutoff]]
                t_minus_indices = [topic_idx for topic_idx, _ in topic_relevance_scores[-cutoff:]]
                
                if len(t_plus_indices) >= 1 and len(t_minus_indices) >= 1:
                    preference_data = {
                        'd': int(doc_idx), 
                        't_plus_indices': [int(x) for x in t_plus_indices], 
                        't_minus_indices': [int(x) for x in t_minus_indices]   
                    }
                    preference_dataset.append(preference_data)
        
        else:
            raise NotImplementedError('Doc-topic preference dataset generation method not supported')
        
        with open(self.doc_topic_preference_dataset_path, 'w', encoding='utf-8') as f:
            for pref_data in preference_dataset:
                f.write(json.dumps(pref_data, ensure_ascii=False) + '\n')
        
        print(f'Created and saved doc-topic preference dataset to: {self.doc_topic_preference_dataset_path}')
        print(f'Generated {len(preference_dataset)} document-topic preference pairs')
        
    def __get_topic_relevance_scores(self, doc_text, top_theta_indices, topic_descriptions):
        relevance_scores = []
        
        for topic_idx in top_theta_indices:
            topic_description = topic_descriptions[topic_idx]
            
            try:
                user_prompt = f"""Document content: "{doc_text}"

                    Topic description: {topic_description}

                    Relevance Score:"""

                # @QUESTION: search about these attributes          
                response = self.__generate_response(
                    system_prompt=cfg.RELEVANCE_GRANDER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_new_tokens=10,
                    temperature=0.3
                )
                
                if response is None:
                    score = 0.5 # Default middle score
                else:
                    score = self.__extract_relevance_score(response)
                
                relevance_scores.append((topic_idx, score))
                
            except Exception as e:
                print(f'ERROR when processing topic {topic_idx}: {e}')
                relevance_scores.append((topic_idx, 0.5)) # Set default score
        
        return relevance_scores
    
    # @QUESTION: about this method
    def __extract_relevance_score(self, response_text):
        """Extract numerical score from LLM response"""
        import re
        
        # Look for numbers in the response (0-10 with optional decimal)
        numbers = re.findall(r'\b([0-9](?:\.[0-9])?|10(?:\.0)?)\b', response_text)
        
        if numbers:
            try:
                score = float(numbers[0])
                # Ensure score is in valid range
                return max(0.0, min(10.0, score))
            except ValueError:
                pass
        
        # If no valid number found, try to interpret text responses
        response_lower = response_text.lower()
        
        if any(word in response_lower for word in ['high', 'very relevant', 'strong', 'excellent']):
            return 8.0
        elif any(word in response_lower for word in ['medium', 'moderate', 'some', 'average']):
            return 5.0
        elif any(word in response_lower for word in ['low', 'weak', 'irrelevant', 'poor']):
            return 2.0
        else:
            return 5.0  # Default middle score
    
    def generate_topic_descriptions(self):
        """Read self.top_words_txt_path and generate description for each topic
        """
        def process_topic_words_str(k, top_words_str):
            top_words = top_words_str.strip().split()
            
            prompt_content = f"Topic {k}: {', '.join(top_words)}"
            
            raw_data = self.__generate_response(
                system_prompt=self.topic_description_system_prompt,
                user_prompt=prompt_content,
                max_new_tokens=150,
                temperature=0.3
            )
            
            if raw_data is None:
                raise RuntimeError(f'Fail to generate response for topic {k}')
            
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

    def __setup_llm(self):
        if self.model_type == 'openai_api':
            print(f'Loading OpenAI LLM: {self.openai_model_name}')
            
            self.openai_client = OpenAI(api_key=self.api_key)
            
            print('Load OpenAI LLM successfully')
        
        elif self.model_type == 'hf_model':
            print(f'Loading Hugging Face LLM: {self.hf_model_name}')
            try: 
                if torch.cuda.is_available():
                    print(f'Loading Hugging Face model: {self.hf_model_name}')
                    
                    if 'mixtral' in self.hf_model_name.lower():
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )

                        self.hf_tokenizer = AutoTokenizer.from_pretrained(
                            self.hf_model_name,
                            token=self.hf_token if self.hf_token else None,
                            trust_remote_code=True
                        )
                        if self.hf_tokenizer.pad_token is None:
                            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

                        self.hf_model = AutoModelForCausalLM.from_pretrained(
                            self.hf_model_name,
                            token=self.hf_token if self.hf_token else None,
                            device_map="auto",
                            quantization_config=quantization_config,
                            torch_dtype=torch.float16,
                            trust_remote_code=True
                        )

                    elif 'gpt2' in self.hf_model_name.lower():
                        self.hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

                        self.hf_model = AutoModelForCausalLM.from_pretrained(
                            "gpt2",
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        ).to("cuda")

                    else:
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(
                            self.hf_model_name,
                            token=self.hf_token if self.hf_token else None,
                            trust_remote_code=True
                        )
                        self.hf_model = AutoModelForCausalLM.from_pretrained(
                            self.hf_model_name,
                            token=self.hf_token if self.hf_token else None,
                            device_map="auto",
                            trust_remote_code=True
                        )

                    self.hf_pipeline = TextGenerationPipeline(
                        model=self.hf_model,
                        tokenizer=self.hf_tokenizer,
                        framework='pt'
                    )
                    
                    print('Load Hugging Face model successfully')
                
                else:
                    print('CUDA not available, fallback to OpenAI API')
                    self.model_type = 'openai_api'
                    self.openai_client = OpenAI(api_key=self.api_key)
            
            except Exception as e:
                print(f'ERROR when load Hugging Face model: {e}')
                print('Fallback to OpenAI API')
                self.model_type = 'openai_api'
                self.openai_client = OpenAI(api_key=self.api_key)
                
        else:
            raise NotImplementedError('Model type not supported')
        
    def __generate_response(self, system_prompt, user_prompt, max_new_tokens=150, temperature=0.3):
        if self.model_type == 'openai_api':
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f'ERROR in OpenAI generation: {e}')
                return None
        
        elif self.model_type == 'hf_model':
            if 'mixtral' in self.hf_model_name.lower():
                prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
            else:
                prompt = f"{system_prompt}\n\n{user_prompt}"
            
            try:
                generation_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'pad_token_id': self.hf_tokenizer.eos_token_id,
                    'eos_token_id': self.hf_tokenizer.eos_token_id,
                    'return_full_text': False,
                    'clean_up_tokenization_spaces': True,
                    'do_sample': True,
                    'temperature': temperature,
                    'top_p': 0.9,
                    'repetition_penalty': 1.1
                }
                    
                response = self.hf_pipeline(prompt, **generation_kwargs)
                
                generated_text = response[0]['generated_text'].strip()
                
                return generated_text
            
            except Exception as e:
                print(f'ERROR in HF model generation: {e}')
                return None
        
        else:
            raise NotImplementedError('Model type not supported')