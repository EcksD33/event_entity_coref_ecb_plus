# TBD

## Introduction
This repo contains experimental code derived from :

<b>"Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution"</b><br/>
Shany Barhom, Vered Shwartz, Alon Eirew, Michael Bugert, Nils Reimers and Ido Dagan. ACL 2019.

Please go to [the original repo](https://github.com/shanybar/event_entity_coref_ecb_plus) for more information on the original code.


## Embeddings used
<b>Character</b> : https://github.com/minimaxir/char-embeddings <br/>
<b>Word2vec</b>  : https://github.com/mmihaltz/word2vec-GoogleNews-vectors <br/>
<b>FastText</b>  : https://pypi.org/project/FastText/ <br/>
<b>GloVe</b>     : https://nlp.stanford.edu/projects/glove/ <br/>
<b>GPT-2</b>     : https://huggingface.co/transformers/model_doc/gpt2.html <br/>
<b>BERT</b>      : https://pypi.org/project/bert-embedding/ <br/>
<b>Elmo</b>      : https://allennlp.org/elmo <br/>

## Branches

### Original model
Original : Code optimizated compared to the [the original repo](https://github.com/shanybar/event_entity_coref_ecb_plus) leading to faster training time

### Ablative models
<b>NoStatic</b>            : Removed GloVe embedding from orignal model <br/>
<b>NoContext</b>           : Removed Elmo embedding from orignal model <br/>
<b>NoChar</b>              : Removed character embedding from orignal model <br/>
<b>noctx-static</b>        : Removed Elmo and GloVe embedding from orignal model <br/>
<b>noctx-static-char</b>   : Removed all embedding from orignal model <br/>

### Comparative models
<b>GPT-2</b>               : Replace Elmo with GPT-2 <br/>
<b>BERT</b>                : Replace Elmo with BERT <br/>
<b>FastText</b>            : Replace GloVe with FastText <br/>
<b>Word2Vec</b>            : Replace GloVe with Word2Vec <br/>

### Comparative ablative models
<b>Onlybert</b>            : Removed GloVe and character embedding from orignal model + Replace Elmo with BERT <br/>
<b>OnlyGPT</b>             : Removed GloVe and character embedding from orignal model + Replace Elmo with GPT-2 <br/>
<b>OnlyELMO</b>            : Removed GloVe and character embedding from orignal model <br/>
<b>onlyfasttext</b>        : Removed GloVe and character embedding from orignal model + Replace GloVe with FastText <br/>
<b>onlyword2vec</b>        : Removed GloVe and character embedding from orignal model + Replace GloVe with Word2Vec <br/>
<b>onlyglove</b>           : Removed GloVe and character embedding from orignal model <br/>

## Contact info
Contact [Judicaël POUMAY](https://github.com/gftvfrbh) at *judicael.poumay@uliege.be* for questions about this repository.
