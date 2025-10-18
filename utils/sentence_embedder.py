'''import os
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import Config as config


class SentenceEmbedder:
    def __init__(self, current_run_dir, args):
        self.model = SentenceTransformer(model_name_or_path=config.BERT_MODEL)
        self.topic_descriptions_path = os.path.join(current_run_dir, 'topic_descriptions.txt')
        self.document_embeddings_path = os.path.join(current_run_dir, 'document_embeddings.npy')
        self.topic_description_embeddings_path = os.path.join(current_run_dir, 'topic_description_embeddings.npy')
        self.documents_path = os.path.join('datasets', args.dataset, 'train_texts.txt')
            
    def embed_topic_descriptions(self):
        with open(self.topic_descriptions_path, 'r') as f:
            topic_descriptions = [line.strip() for line in f.readlines()]
            
        topic_description_embeddings = self.model.encode(topic_descriptions)
        np.save(self.topic_description_embeddings_path, topic_description_embeddings)
        
        print(f"Topic description embeddings saved to: {self.topic_description_embeddings_path}")
        print(f"Embeddings shape: {topic_description_embeddings.shape}")
    
    def embed_documents(self):
        with open(self.documents_path, 'r') as f:
            documents = [line.strip() for line in f.readlines()]
            
        document_embeddings = self.model.encode(documents)
        np.save(self.document_embeddings_path, document_embeddings)
        
        print(f"Document embeddings saved to: {self.document_embeddings_path}")
        print(f"Embeddings shape: {document_embeddings.shape}")
    
    def load_document_embeddings(self):
        document_embeddings = np.load(self.document_embeddings_path)
        return document_embeddings
    
    def load_topic_description_embeddings(self):
        topic_description_embeddedings = np.load(self.topic_description_embeddings_path)
        return topic_description_embeddedings'''