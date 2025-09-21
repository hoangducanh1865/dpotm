import numpy as np
import logging


def print_topic_words(beta, vocab, num_top_words):
    logger = logging.getLogger('main')
    topic_str_list = list()
    topic_indices_list = list()
    for i, topic_dist in enumerate(beta):
        # Get indices of top words in beta matrix (sorted by probability descending)
        top_word_indices = np.argsort(topic_dist)[:-(num_top_words + 1):-1]
        topic_words = np.array(vocab)[top_word_indices]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        topic_indices_list.append(top_word_indices.tolist())
        print('Topic {}: {}'.format(i, topic_str))
        logger.info('Topic {}: {}'.format(i, topic_str))
    return topic_str_list, topic_indices_list