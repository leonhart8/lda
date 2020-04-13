"""
Module used to implement a preprocessing pipeline used to prepare data
for LDA

authors :
    - Mathis Demay 
    - Luqman Ferdjani
"""
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import numpy as np

class Preprocessing():
    """
    The class which encapsulates the data pre-processing pipeline
    used for LDA on the newsgroup dataset, steps include :
        -   Tokenization of text
        -   Removal of stop words
        -   Lemmatization
        -   Building count of occurrences of words in each doc
        -   Building an index of all words
    """
    

    def doc_preproc(self, doc):
        """
        Applies the preprocessing pipeline to a single document
        """
        stemmer = SnowballStemmer("english")
        processed = []
        tokens = gensim.utils.simple_preprocess(doc)
        for token in tokens:
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                processed.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
        return processed


    def corpus_preproc(self, corpus):
        """
        Applies the preprocessing pipeline to an entire corpus
        """
        nltk.download("wordnet")
        processed = []
        for doc in corpus:
            processed.append(self.doc_preproc(doc))
        return processed


    def build_bow(self, proc_corpus):
        """
        Builds a bow of each doc from the processed corpus
        """
        d = gensim.corpora.Dictionary(proc_corpus)
        d.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
        bow = [d.doc2bow(doc) for doc in proc_corpus]
        return d, bow


if __name__ == '__main__':
    """
    Example of application using newsgroups
    """
    from sklearn.datasets import fetch_20newsgroups
    pp = Preprocessing()
    newsgroups = fetch_20newsgroups() # raw data
    proc_corpus = pp.corpus_preproc(newsgroups["data"]) # preprocess it
    print(proc_corpus[20:23])
    d, bow = pp.build_bow(proc_corpus)
    print(d)
    print(len(d.token2id))
    print(bow[20])