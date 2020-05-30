import logging

import numpy as np
from bert_embedding import BertEmbedding

logger = logging.getLogger(__name__)


class BERTEmbedding(object):
    '''
    A wrapper class for the bert_embedding
    '''
    def __init__(self):
        logger.info('Loading BERT Embedding module')
        self.embedder = BertEmbedding()
        logger.info('BERT Embedding module loaded successfully')

    def get_embedding(self, sentence):
        result = bert_embedding([sentence])
        return result[0]



