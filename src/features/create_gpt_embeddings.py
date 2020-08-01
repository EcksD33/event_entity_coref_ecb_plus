from pytorch_transformers  import GPT2Tokenizer, GPT2Model
import torch
import logging
import numpy as np


logger = logging.getLogger(__name__)


class GPT2Embedding(object):
    '''
    A wrapper class for the bert_embedding
    '''
    def __init__(self):
        self.cuda0 = torch.device('cuda:0')
        logger.info('Loading GPT Embedding module')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = " "
        self.model = GPT2Model.from_pretrained('gpt2')
        logger.info('GPT Embedding module loaded successfully')
        

    def get_embedding(self, sentence):
        tokenized_sent = sentence.get_tokens_strings()
        inputs = self.tokenizer.encode(sentence.get_raw_sentence())
        result = self.model(torch.tensor(inputs))[0].detach().numpy()
        tokens = []
        for ids in inputs:
            tokens.append(self.tokenizer.decoder[ids].replace("Ġ",""))
        
        out = [np.zeros(768) for i in range(len(tokenized_sent))]
        for i,toks in enumerate(tokenized_sent):
            count = 0
            for j,gptToks in enumerate(tokens):
                if(gptToks is None):
                    continue
                if(toks == gptToks):
                    out[i] = result[j]
                    break        
                if(toks in gptToks):
                    out[i] = result[j]
                    break          
                if(gptToks in toks):
                    out[i] += result[j]
                    count += 1
            if(count > 0):
                out[i]/= count
        out = [x/100 for x in out]
        return out
        