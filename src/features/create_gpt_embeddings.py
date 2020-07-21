from transformers import GPT2Tokenizer, GPT2Model
import torch



logger = logging.getLogger(__name__)


class GPT2Embedding(object):
    '''
    A wrapper class for the bert_embedding
    '''
    def __init__(self):
        logger.info('Loading GPT Embedding module')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        logger.info('GPT Embedding module loaded successfully')

    def get_embedding(self, sentence):
        tokenized_sent = sentence.get_tokens_strings()
        inputs = tokenizer(sentence.get_raw_sentence(), return_tensors="pt")
        result = model(**inputs)[0][0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        
        out = [np.zeros(768) for i in range(len(tokenized_sent))]
        for i,toks in enumerate(tokenized_sent):
            count = 0
            for j,gptToks in enumerate(tokens):
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
        
        return out



