import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.configs import Configs as cfg 


class PreferenceDatasetCreator:
    def __init__(self, dir_path):
        load_dotenv()
        self.preference_dataset_path = os.path.join(dir_path, 'preference_dataset.jsonl')
        self.top_words_10_path = os.path.join(dir_path, 'top_words_10.jsonl')
        self.top_words_15_path = os.path.join(dir_path, 'top_words_15.jsonl')
        self.top_words_20_path = os.path.join(dir_path, 'top_words_20.jsonl')
        self.top_words_25_path = os.path.join(dir_path, 'top_words_25.jsonl')
        self.model = cfg.LLM_MODEL
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = cfg.SYSTEM_PROMPT

    def create(self):
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
                    {"role": "system", "content": self.system_prompt},
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
            
        with open(self.top_words_25_path, 'r', encoding='utf-8') as infile, open(self.preference_dataset_path, 'w', encoding='utf-8') as outfile:
            for k, line in enumerate(infile):
                data = process_line(k, line)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # Debug
        print(f'Created and saved preference dataset to: {self.preference_dataset_path}')
