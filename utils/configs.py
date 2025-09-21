import torch


class Configs:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    CHECKPOINT_PATH = ''
    '''
    results/ECRTM/StackOverflow/2025-09-21_01-38-04/checkpoints/checkpoint_epoch_500.pth
    results/ECRTM/20NG/2025-09-21_23-00-30/checkpoints/checkpoint_epoch_500.pth
    results/ECRTM/WOS_vocab_5k/2025-09-22_00-04-35/checkpoints/checkpoint_epoch_500.pth
    '''
    
    LLM_MODEL = 'gpt-4o-mini'
    SYSTEM_PROMPT = """You are a text classifier.  
Your task is to analyze a list of words with their associated indices in the beta matrix.

For each topic:
1. Identify the main topic that most of the words are related to.  
2. Describe that topic briefly in a few English words.  
3. Return only one JSON object in the following format:

{
  "k": <topic_index>,
  "topic": "<short English description>",
  "w_plus_indices": [<beta_indices of words related to the main topic>],
  "w_minus_indices": [<beta_indices of words not related to the main topic>]
}

Notes:
- Use the beta matrix indices provided with each word, not the position in the list.
- "w_plus_indices" should contain beta indices of words that are coherent with the main topic.  
- "w_minus_indices" should contain beta indices of words that are unrelated or noisy.  
- Do not include explanations, only output the JSON object.
"""