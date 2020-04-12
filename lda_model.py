"""
Module used to implement Latent Dirichlet Allocation (LDA) a generative probabilistic
modelling for word and topic generation inside documents.

This module makes use of a paper from the Journal of Machine Learning Research 3 (2003) 993-1022
by David M. Blei, Andrew Y. Ng ad Michael I. Jordan which details the LDA model in itself,
the variational methods for inference it employs, and applications. All credit for the methods
employed belong to them.

authors :
    - Mathis Demay 
    - Luqman Ferdjani
"""
import numpy as np
from preprocessing import Preprocessing
from scipy.special import digamma, loggamma

class LDA():
    """
    Class which encapsulates all information relevant to Latent Dirichlet Allocation.
    Can be fitted to collection of documents which are summarized by their word
    occurrences.
    Uses variational methods to estimate parameters linked to topic and
    word distribtions and uses these and for inference
    """


    def __init__(self, nb_topics, bow, index, alpha, set_alpha=False):
        """
        Constructor of a LDA model instance in order to predict topic probabilities
        for documents comprised of nb_terms distinct words at maximum

        :param nb_topics: int, amount of topics inside corpus of documents
        :param nb_terms: int, number of distinct terms in total throughout the corpus
        :param alpha: a vector of nb_topics concentration parameters of a Dirichlet

        :return: LDA model with set fields according to what is said above
        """
        self.nb_topics = nb_topics
        self.bow = bow
        self.index = index

        #If the user decides to set alpha we take their value
        #Else we set a uniform prior over topic distribution 
        if set_alpha:
            self.alpha = alpha
        else:
            self.alpha = 1

        #Computing maximum number of unique terms inside a document
        #Necessary for allocation of the variational parameter phi
        maximum = 0

        for b in bow:
            if len(b) > maximum:
                maximum = len(b)

        self.doc_max_length = maximum

        #Computing number of documents
        self.nb_docs = len(bow)

        #Computing number of unique unigrams in total
        self.nb_terms = len(index.token2id)

        #Beta parameter as described in paper, a k x V matrix with :
        #   - k topics
        #   - V words
        self.beta = np.zeros(shape=(nb_topics, self.nb_terms))

        #Variational parameters
        self.gamma = None
        self.phi = None


    def nb_terms_doc(self, doc_idx):
        """
        Counts the number of terms inside a document
        """
        cpt_words, cpt_d_words =  0, len(self.bow[doc_idx])
        for t in self.bow[doc_idx]:
            cpt_words += t[1]
        return cpt_words, cpt_d_words


    def check_word_id_in_doc(self, word_id, doc_idx):
        """
        Check if the word of id word_id is in the document
        of id doc_idx
        """
        dic = dict(self.bow[doc_idx])
        return word_id in dic
            

    def estimation(self, em_threshold=10e-5, e_threshold=10e-8, max_iter=None):
        """
        Estimation of the alpha and beta parameters of the LDA model
        Uses a EM algorithm which functions the following way :
            - E step by computing best variational parameters
            - M step in order to use these parameters to estimate alpha
            and beta which maximize the expectation of the likelihood
        """
        has_converged = False 
        nb_iter = 0

        #Initializing variational parameters
        gamma = np.zeros((self.nb_docs, self.nb_topics))
        phi = np.zeros((self.nb_docs, self.doc_max_length, self.nb_topics))
        
        for d in range(self.nb_docs):
            #total count of words inside document
            #and total count of distinct words inside document
            nb_words_doc, nb_d_words_doc = self.nb_terms_doc(d)
            for n in range(nb_d_words_doc):
                phi[d][n] = 1 / self.nb_topics
            gamma[d] = self.alpha + nb_words_doc / self.nb_topics

        #EM algorithm
        while (not has_converged and (max_iter is None or nb_iter < max_iter)):

            print("Iteration:", nb_iter)
            likelihood = 0

            #Expectation computation part : optimization of the variational parameters
            print("E-Step")

            for d in range(self.nb_docs):
                if (d % 1000 == 0):
                    print("E-step through", d, "documents")

                likelihood += self.inference_doc(d, gamma[d], phi[d], conv=e_threshold)

            #Maximization of the expectation : maximize lower bound of the log likelihood of the variational distribution
            print("M-Step")

            #Estimating beta
            for d in range(self.nb_docs):
                _, nb_d_words_doc = self.nb_terms_doc(d)
                for k in range(self.nb_topics):
                    for v in range(self.nb_terms):
                        if self.check_word_id_in_doc(v, d):
                            for n in range(nb_d_words_doc):
                                self.beta[k][v] += phi[d][n][k]
                    #Normalizing multinomials
                    if np.sum(self.beta[k]) > 0:
                        self.beta[k] /= np.sum(self.beta[k])

            print(np.max(self.beta))

            #Estimating alpha with Newton-Raphson
            # TO DO

            if (nb_iter > 0):
                diff = (prev_likelihood - likelihood) / prev_likelihood
                if diff < em_threshold:
                    has_converged = True

            prev_likelihood = likelihood

            print("iteraton", nb_iter, likelihood)

            nb_iter += 1

        self.gamma = gamma
        self.phi = phi

        return self.beta, gamma, phi


    def inference_doc(self, doc_idx, gamma_doc, phi, conv=10e-8, max_iter=None):
        """
        Variational inference algorithm described in the paper
        Uses gamma and phi, two variational parameters used to approximate
        the distribution of the parameters alpha and beta knowing our data (=corpus)
        :param self: this LDA model
        :param doc_id: int, index of the document on which to perform
        """
        has_converged = False
        nb_iter = 0
        _, nb_d_words_doc = self.nb_terms_doc(doc_idx)

        #Now estimating gamma and phi
        while not has_converged and ((max_iter is None) or (nb_iter < max_iter)):
            print("Iteration", nb_iter, "of variational parameters estimation")
            for i in range(nb_d_words_doc):
                for j in range(self.nb_topics):
                    word_index = self.bow[doc_idx][i][0]
                    phi[i][j] = self.beta[j][word_index] * np.exp(digamma(gamma_doc[j]))
                if np.sum(phi[i]) > 0:
                    phi[i] /= (np.sum(phi[i]))
            sum_phi = np.zeros(self.nb_topics)
            for i in range(nb_d_words_doc):
                sum_phi += phi[i]
            gamma_doc = self.alpha + sum_phi

            likelihood = self.likelihood(doc_idx, nb_d_words_doc, gamma_doc, phi)
            print(likelihood)

            if nb_iter > 0:
                diff = (prev_likelihood - likelihood) / prev_likelihood
                if diff < conv:
                    has_converged = True
                
            prev_likelihood = likelihood

            nb_iter += 1

        return likelihood


    def likelihood(self, doc_idx, nb_d_words_doc, gamma_doc, phi):
        """
        Computation of the likelihood lower bound as described in the paper
        """
        term_1 = loggamma(np.sum(self.alpha)) - np.sum(loggamma(self.alpha)) +\
            np.sum((self.alpha - 1) * (digamma(gamma_doc)) - digamma(np.sum(gamma_doc)))

        term_2 = 0.0

        for i in range(nb_d_words_doc):
            for j in range(self.nb_topics):
                term_2 += phi[i][j] * (digamma(gamma_doc[j]) - digamma(np.sum(gamma_doc)))

        term_3 = 0.0

        for i in range(nb_d_words_doc):
            for j in range(self.nb_topics):
                for k in range(self.nb_terms):
                    if self.check_word_id_in_doc(k, doc_idx) and self.beta[j][k] != 0:
                        term_3 += phi[i][j] * np.log(self.beta[j][k])

        term_4 = loggamma(np.sum(gamma_doc)) + np.sum(loggamma(gamma_doc)) -\
            np.sum((gamma_doc - 1) * (digamma(gamma_doc[j]) - digamma(np.sum(gamma_doc))))

        term_5 = 0.0

        for i in range(nb_d_words_doc):
            for j in range(self.nb_topics):
                if phi[i][j] != 0:
                    term_5 += phi[i][j] * np.log(phi[i][j])

        return term_1 + term_2 + term_3 - term_4 - term_5


if __name__ == "__main__":
    """
    Example of application using newsgroups
    """
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    pp = Preprocessing()

    index, bow = pp.build_bow(pp.corpus_preproc(train["data"]))

    lda = LDA(5, bow, index, alpha=50, set_alpha=True)

    lda.estimation()