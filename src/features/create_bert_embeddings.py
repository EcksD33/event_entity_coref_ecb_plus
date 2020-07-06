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
        self.embedder = BertEmbedding(model='bert_24_1024_16',max_seq_length =50)
        logger.info('BERT Embedding module loaded successfully')

    def get_embedding(self, sentence):
        tokenized_sent = sentence.get_tokens_strings()
        result = self.embedder([sentence.get_raw_sentence()])[0]
        
        out = [np.zeros(1024) for i in range(len(tokenized_sent))]
        for i,toks in enumerate(tokenized_sent):
            count = 0
            for j,bertToks in enumerate(result[0]):
                if(toks.lower() == bertToks):
                    out[i] = result[1][j]
                    break        
                if(toks.lower() in bertToks):
                    out[i] = result[1][j]
                    break          
                if(bertToks.lower() in toks):
                    out[i] += result[1][j]
                    count += 1
            if(count > 0):
                out[i]/= count
        
        return out



