import torch


class Configs:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    CHECKPOINT_PATH = 'results/ECRTM/StackOverflow/2025-09-21_01-38-04/checkpoints/checkpoint_epoch_500.pth'
    
    LLM_MODEL = 'gpt-4o-mini'
    SYSTEM_PROMPT = """You are a text classifier.  
Your task is to analyze a line of 25 words.  

For each line:
1. Identify the main topic that most of the words are related to.  
2. Describe that topic briefly in a few English words.  
3. Return only one JSON object in the following format:

{
  "k": <line_index>,
  "topic": "<short English description>",
  "w_plus_indices": [<indices of words related to the main topic>],
  "w_minus_indices": [<indices of words not related to the main topic>]
}

Notes:
- Indices start at 0.  
- "w_plus_indices" corresponds to the majority topic.  
- "w_minus_indices" corresponds to words unrelated or minority.  
- Do not include explanations, only output the JSON object.
"""