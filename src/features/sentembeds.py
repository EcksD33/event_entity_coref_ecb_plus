"""Module used to test different sentence embeddings on top of Bahrom's model.
Do note that some of these require some setup and different dependencies.
"""


from numpy import dtype


def get_embedder(which: str):
    which = which.lower().strip()

    # https://github.com/pdrm83/sent2vec, [768]
    # can be installed through pip: pip install sent2vec
    # requires:
    #   gensim
    #   numpy
    #   spacy
    #   transformers
    #   pytorch
    if which == "pdrm83":
        import numpy as np
        from features.pdrm83.vectorizer import Vectorizer

        model = Vectorizer()

        def embedder(X):
            model.bert(X)
            return np.array(model.vectors)

        return embedder

    # https://github.com/ryankiros/skip-thoughts, [4800]
    # needs to be converted to python3
    # requires:
    #   Theano 0.7
    #   NumPy
    #   SciPy
    #   scikit-learn
    #   NLTK 3
    elif which == "skipthoughts":
        import features.skipthoughts3.skipthoughts as skipthoughts

        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)

        def embedder(X):
            return encoder.encode(X)

        return embedder

    # https://github.com/facebookresearch/InferSent, [4096]
    # requires:
    #   Pytorch (recent version)
    #   NLTK >= 3
    elif which == "infersent":
        import torch
        import nltk
        from features.infersent.models import InferSent
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            from importlib import reload
            nltk.download('punkt')
            nltk = reload(nltk)

        V = 2
        MODEL_PATH = 'res/pretrained/infersent/infersent%s.pkl' % V
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))

        W2V_PATH = 'res/pretrained/fastText/crawl-300d-2M.vec'
        infersent.set_w2v_path(W2V_PATH)
        infersent.build_vocab_k_words(K=100000)

        def embedder(X):
            return infersent.encode(X, tokenize=True)

        return embedder

    # https://github.com/celento/sent2vec, [upto 700]
    # requires:
    #   Cython
    #   Microsoft Visual C++ (MSVC 140)
    # then run:
    #   cd src/features/celento_sent2vec
    #   pip install --global-option build_ext --global-option --compiler=mingw32 .
    # see https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
    # or  https://stackoverflow.com/a/19346426
    elif which == "celento":
        import sent2vec
        import re
        import nltk
        import numpy as np

        def tokenize(tknzr, sentence, to_lower=True):
            """Arguments:
                - tknzr: a tokenizer implementing the NLTK tokenizer interface
                - sentence: a string to be tokenized
                - to_lower: lowercasing or not
            """
            sentence = sentence.strip()
            sentence = ' '.join([format_token(x) for x in tknzr(sentence)])
            if to_lower:
                sentence = sentence.lower()
            sentence = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', sentence)  # replace urls by <url>
            sentence = re.sub(r'(\@[^\s]+)', '<user>', sentence)  # replace @user268 by <user>
            filter(lambda word: ' ' not in word, sentence)

            return sentence

        def format_token(token):
            """"""
            if token == '-LRB-':
                token = '('
            elif token == '-RRB-':
                token = ')'
            elif token == '-RSB-':
                token = ']'
            elif token == '-LSB-':
                token = '['
            elif token == '-LCB-':
                token = '{'
            elif token == '-RCB-':
                token = '}'
            return token

        def tokenize_sentences(tknzr, sentences, to_lower=True):
            """Arguments:
                - tknzr: a tokenizer implementing the NLTK tokenizer interface
                - sentences: a list of sentences
                - to_lower: lowercasing or not
            """
            return [tokenize(tknzr, s, to_lower) for s in sentences]

        model = sent2vec.Sent2vecModel()
        model.load_model("res/pretrained/celento/torontobooks_unigrams.bin")
        tknzr = nltk.word_tokenize

        def embedder(X):
            embed = model.embed_sentences(tokenize_sentences(tknzr, X))
            return np.array(embed).astype(np.float32)

        return embedder

    raise ValueError(f"Argument '{which}' not recognized or not implemented.")
