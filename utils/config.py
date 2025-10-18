import torch
import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument(
        "--dataset", type=str, help="dataset name", default="BBC_new"
    )  # ['BBC_new', '20NG', 'WOS_vocab_5k']
    parser.add_argument(
        "--plm_model", type=str, help="plm model name", default="all-mpnet-base-v2"
    )


def add_logging_argument(parser):
    parser.add_argument("--wandb_prj", type=str, default="topmost")


def add_model_argument(parser):
    parser.add_argument("--model", type=str, default="ECRTM")  # ['ECRTM']
    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--num_top_words", type=int, default=20)
    parser.add_argument("--num_groups", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim_1", type=int, default=384)
    parser.add_argument("--hidden_dim_2", type=int, default=384)
    parser.add_argument("--theta_temp", type=float, default=1.0)
    parser.add_argument("--DT_alpha", type=float, default=3.0)
    parser.add_argument("--TW_alpha", type=float, default=2.0)

    parser.add_argument("--weight_GR", type=float, default=1.0)
    parser.add_argument("--alpha_GR", type=float, default=5.0)
    parser.add_argument("--weight_InfoNCE", type=float, default=50.0)
    parser.add_argument("--beta_temp", type=float, default=0.2)
    parser.add_argument(
        "--weight_ECR", type=float, default=350.0
    )  # [100.0, 350.0] # Use 350.0 for better TD, use 100.0 for better TC
    parser.add_argument(
        "--use_pretrainWE",
        action="store_true",
        default=False,
        help="Enable use_pretrainWE mode",
    )
    parser.add_argument("--weight_dpo", type=float, default=0.5)
    parser.add_argument("--weight_reg", type=float, default=0.5)


def add_wete_argument(parser):
    parser.add_argument(
        "--glove", type=str, default="glove.6B.100d.txt", help="embedding model name"
    )
    parser.add_argument("--wete_beta", type=float, default=0.5)
    parser.add_argument("--wete_epsilon", type=float, default=0.1)
    parser.add_argument("--init_alpha", action="store_true", default=False)


def add_training_argument(parser):
    parser.add_argument("--use_kaggle", type=int, default=0)
    parser.add_argument("--use_colab", type=int, default=0)
    parser.add_argument(
        "--llm_type", type=str, default="openai_api"
    )  # ['openai_api', 'hf_model']
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    """parser.add_argument('--finetune_lr', type=float, default=0.002, # [0.0001, 0.0005, 0.001, 0.002]
                        help='fine-tune learning rate')"""
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="learning rate scheduler, dont use if not needed, \
                            currently support: step",
        default="StepLR",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=125,
        help="step size for learning rate scheduler",
    )
    parser.add_argument("--finetune_beta", type=int, default=0)
    parser.add_argument("--finetune_theta", type=int, default=0)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to checkpoint file to resume training",
    )
    parser.add_argument(
        "--loss_topic_word_dpo_calculation_method", type=str, default="multiply"
    )  # ['multiply', 'hard_negative', 'hard_positive', 'combined_hard']
    parser.add_argument(
        "--loss_doc_topic_dpo_calculation_method",
        type=str,
        default="multiply",
        choices=["multiply", "hard_negative", "hard_positive", "combined_hard"],
    )
    parser.add_argument("--use_jaccard", action="store_true", default=False)
    parser.add_argument(
        "--loss_dpo_topic_word_type", type=str, default="bradley_terry"
    )  # ['bradley_terry', 'plackett_luce']
    parser.add_argument(
        "--loss_dpo_doc_topic_type", type=str, default="plackett_luce"
    )  # ['bradley_terry', 'plackett_luce']
    parser.add_argument(
        "--generate_topic_descriptions_method", type=str, default="openai"
    )  # ['openai', 'hf']
    parser.add_argument(
        "--topic_word_dataset_generate_method", type=str, default="openai"
    )  # ['openai', 'hf']
    parser.add_argument(
        "--doc_topic_dataset_generate_method", type=str, default="hf"
    )  # ['openai', 'hf']
    parser.add_argument("--num_docs_per_call", type=int, default=1)


def add_eval_argument(parser):
    parser.add_argument("--tune_SVM", action="store_true", default=False)


def save_config(args, path):
    with open(path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def load_config(path):
    args = argparse.Namespace()
    with open(path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            if value.isdigit():
                if value.find(".") != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LLM_MODEL = "gpt-4o-mini"
    BERT_MODEL = "all-MiniLM-L6-v2"
    HF_MODEL = "Qwen/Qwen3-32B"

    TOPIC_WORD_SYSTEM_PROMPT = """You are a text classifier.  
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
- Do not include explanations, only output the JSON object."""

    TOPIC_DESCRIPTION_SYSTEM_PROMPT = """You are a topic analysis expert.
Your task is to analyze a list of top words from a topic and provide a clear, concise description.

For each topic:
1. Analyze the semantic relationship between the words
2. Identify the main theme or subject domain  
3. Create a concise topic name (2-4 words)
4. Write a descriptive sentence explaining what this topic represents

Return only one JSON object in the following format:

{
    "topic_name": "<concise topic name>",
    "description": "<detailed description sentence>",
    "key_words": ["<most relevant words from the list>"]
}

Notes:
- Keep the topic name short and descriptive (2-4 words)
- Make the description informative but concise
- Include 3-5 most relevant words in key_words array
- Do not include explanations, only output the JSON object."""

    @staticmethod
    def get_doc_topic_prompt(topic_descriptions_str, batch_docs):
        prompt = f"""Sort the given topics by relevance for each document (most relevant first).\n\n"""

        # Add each document with its topics
        for doc_data in batch_docs:
            doc_idx = doc_data["doc_idx"]
            doc_text = doc_data["doc_text"]
            top_5_topic_indices = doc_data["top_5_topic_indices"]

            prompt += f"Document {doc_idx}: {doc_text}\n"
            prompt += (
                f"Sort these topics by relevance: {top_5_topic_indices.tolist()}\n\n"
            )

        prompt += f"""Topic descriptions:
{topic_descriptions_str}

Instructions:
- For each document, ONLY reorder its given topic numbers (do not add or remove any)
- Return exactly {len(batch_docs)} lines
- Each line: the same 5 numbers reordered by relevance (most relevant first)
- Use ONLY the numbers provided for each document
- Format: doc1_sorted_topics\\ndoc2_sorted_topics\\n...

Example:
If given topics [12, 5, 33, 41, 9], return them sorted: 33, 9, 12, 5, 41
If given topics [7, 22, 15, 3, 28], return them sorted: 15, 3, 7, 28, 22

Your sorted topics:
"""

        return prompt
