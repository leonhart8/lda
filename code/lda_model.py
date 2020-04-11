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
from scipy.special import digamma, loggamma

class lda():
    """
    Class which encapsulates all information relevant to Latent Dirichlet Allocation.
    Can be fitted to collection of documents which are summarized by their word
    occurrences.
    Uses variational methods to estimate parameters linked to topic and
    word distribtions and uses these and for inference
    """

    def __init__(self, nb_topics, nb_terms, alpha, set_alpha=False):
        """
        Constructor of a LDA model instance in order to predict topic probabilities
        for documents comprised of nb_terms distinct words at maximum

        :param nb_topics: int, amount of topics inside corpus of documents
        :param nb_terms: int, number of distinct terms in total throughout the corpus
        :param alpha: a vector of nb_topics concentration parameters of a Dirichlet

        :return: LDA model with set fields according to what is said above
        """
        self.nb_topics = nb_topics
        self.nb_terms = nb_terms

        #If the user decides to set alpha we take their value
        #Else we set a uniform prior over topic distribution 
        if set_alpha:

            assert alpha.shape[0] == nb_topics
            self.alpha = alpha

        else:

            self.alpha = np.ones(shape=nb_topics)

        self.beta = np.zeros(shape=(nb_topics, nb_terms))


    def inference(self, doc, nb_words_doc, gamma_doc, phi, conv=10e-8, max_iter=None):
        """
        Variational inference algorithm described in the paper
        Uses gamma and phi, two variational parameters used to approximate
        the distribution of the parameters alpha and beta knowing our data (=corpus)
        :param self: this LDA model
        :param doc_id: int, index of the document on which to perform
        """
        has_converged = False
        prev_likelihood = 1

        # Initialization of variational parameters as in the paper
        for i in range(self.nb_topics):
            gamma_doc[i] = self.alpha[i] + (nb_words_doc / self.nb_topics)
            for j in range(nb_words_doc):
                phi[i][j] = 1 / self.nb_topics


        nb_iter = 0
                
        word_index = 

        #Now estimating gamma and phi
        while not has_converged and ((max_iter is None) or (nb_iter < max_iter)):
            for i in range(nb_words_doc):
                for j in range(self.nb_topics):
                    phi[i][j] = self.beta[j][word_index] * np.exp(digamma(gamma_doc[i]))
                phi[i] /= (np.sum(phi[i]))
            sum_phi = np.zeros(self.nb_topics)
            for i in range(nb_words_doc):
                sum_phi += phi[i]
            gamma_doc = alpha + sum_phi

            likelihood = self.likelihood(doc, nb_words_doc, gamma_doc, phi)

            diff = (prev_likelihood - likelihood) / prev_likelihood
            prev_likelihood = likelihood

            if diff < conv:

                has_converged = True

            print("Likelihood :", likelihood, "Increase rate :", diff, "iteration :", nb_iter)

            nb_iter += 1

        return likelihood


    def likelihood(self, doc, nb_words_doc, gamma_doc, phi):
        """
        Computation of the likelihood as described in the paper
        """
        term_1 = loggamma(np.sum(alpha)) - np.sum(loggamma(alpha)) +\
            + np.sum((alpha - 1) * (digamma(gamma_doc)) - digamma(np.sum(gamma_doc)))

        term_2 = 0

        for i in range(nb_words_doc):
            for j in range(self.nb_topics):
                term_2 += phi[i][j] * (digamma(gamma_doc) - digamma(np.sum(gamma_doc)))

        term_3 = 0

        for i in range(nb_words_doc):
            for j in range(self.nb_topics):
                for k in range(self.nb_terms):
                    term_3 += phi[i][j] * count_word * np.log(beta[j][k])

        term_4 = loggamma(np.sum(gamma_doc)) + np.sum(loggamma(gamma_doc)) -\
            np.sum((gamma_doc - 1) * (digamma(gamma_doc) - digamma(np.sum(gamma_doc))))

        term_5 = 0

        for i in range(nb_words_doc):
            for j in range(self.nb_topics):
                term_5 += phi[i][j] * np.log(phi[i][j])

        return term_1 + term_2 + term_3 - term_4 - term_5

