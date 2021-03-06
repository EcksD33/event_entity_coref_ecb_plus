import logging

import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

logger = logging.getLogger(__name__)


class ElmoEmbedding(object):
    '''
    A wrapper class for the ElmoEmbedder of Allen NLP
    '''
    def __init__(self, options_file, weight_file):
        logger.info('Loading Elmo Embedding module')
        # scalar_mix_parameters are the weights used to average the three layers
        self.embedder = Elmo(options_file, weight_file, 1, dropout=0,
                             scalar_mix_parameters=[1, 1, 1])
        logger.info('Elmo Embedding module loaded successfully')

    def get_embedding(self, sentence):
        '''
        This function gets a sentence object and returns and ELMo embeddings of
        each word in the sentences (specifically here, we average over the 3 ELMo layers).
        :param sentence: a sentence object
        :return: the averaged ELMo embeddings of each word in the sentences
        '''
        tokenized_sent = sentence.get_tokens_strings()
        output = self.embedder(batch_to_ids([tokenized_sent]))
        # the next line is no longer needed since this is done in __init__
        # output = np.average(embeddings, axis=0)

        # first [0] is for batch size (a single sentence = 1 batch)
        # second [0] is for the number of timesteps
        return output["elmo_representations"][0][0].detach().numpy()
