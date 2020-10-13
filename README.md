# TBD

## Introduction
This repo contains experimental code derived from :

<b>"Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution"</b><br/>
Shany Barhom, Vered Shwartz, Alon Eirew, Michael Bugert, Nils Reimers and Ido Dagan. ACL 2019.

Please go to [the original repo](https://github.com/shanybar/event_entity_coref_ecb_plus) for more information on the original code.


## Embeddings used
Character : https://github.com/minimaxir/char-embeddings
Word2vec  : https://github.com/mmihaltz/word2vec-GoogleNews-vectors
FastText  : https://pypi.org/project/FastText/
GloVe     : https://nlp.stanford.edu/projects/glove/
GPT-2     : https://huggingface.co/transformers/model_doc/gpt2.html
BERT      : https://pypi.org/project/bert-embedding/
Elmo      : https://allennlp.org/elmo

## Branches

### Original model
Original : Code optimizated compared to the [the original repo](https://github.com/shanybar/event_entity_coref_ecb_plus) leading to faster training time

### Ablative models
NoStatic            : Removed GloVe embedding from orignal model
NoContext           : Removed Elmo embedding from orignal model
NoChar              : Removed character embedding from orignal model
noctx-static        : Removed Elmo and GloVe embedding from orignal model
noctx-static-char   : Removed all embedding from orignal model

### Comparative models
GPT-2               : Replace Elmo with GPT-2
BERT                : Replace Elmo with BERT     
FastText            : Replace GloVe with FastText 
Word2Vec            : Replace GloVe with Word2Vec

### Comparative ablative models
Onlybert            : Removed GloVe and character embedding from orignal model + Replace Elmo with BERT
OnlyGPT             : Removed GloVe and character embedding from orignal model + Replace Elmo with GPT-2            
OnlyELMO            : Removed GloVe and character embedding from orignal model             
onlyfasttext        : Removed GloVe and character embedding from orignal model + Replace GloVe with FastText                
onlyword2vec        : Removed GloVe and character embedding from orignal model + Replace GloVe with Word2Vec                  
onlyglove           : Removed GloVe and character embedding from orignal model          

## Contact info
Contact [Judicaël POUMAY](https://github.com/gftvfrbh) at *judicael.poumay@uliege.be* for questions about this repository.
