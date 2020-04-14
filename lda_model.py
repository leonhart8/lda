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
from scipy.special import digamma, loggamma, polygamma, logsumexp

class LDA():
    """
    Class which encapsulates all information relevant to Latent Dirichlet Allocation.
    Can be fitted to a collection of documents which are summarized by their word
    occurrences.
    Uses variational methods to estimate parameters linked to topic and
    word distribtions and uses these and for inference.
    """


    def __init__(self, nb_topics, bow, index, alpha, set_alpha=False):
        """
        Constructor of a LDA model instance in order to predict topic probabilities
        for documents comprised of nb_terms distinct words at maximum

        :param nb_topics: int, amount of topics inside corpus of documents
        :param index: an index of words
        :param bow: a bag of words, a list of lists of tuples containing (key, occurrences)
        :param alpha: a vector of nb_topics concentration parameters of a Dirichlet
        used to generate theta. High values of alpha means that our prior
        knowledge indicates that topics are uniformly distributed, low alpha
        will favor sparse topic multinomials
        :param set_alpha: if a value of alpha is given by user, must set flag to Trues

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
        #Of note : we are manipulating log probabilities and not
        #Probabilities themselves, hence the zero values
        self.beta = np.zeros(shape=(nb_topics, self.nb_terms))

        #Variational parameters
        self.gamma = None
        self.phi = None

        #Topic word total is necessary for normalizing lines along the beta matrix
        #Topic count per word is also neessary to updating coefficients of the beta
        #Matrix using word counts
        self.topic_tot = np.zeros(self.nb_topics)
        self.topic_word = np.zeros((self.nb_topics, self.nb_terms))


    def nb_terms_doc(self, doc_idx):
        """
        Count the number of terms inside a document.
        
        :param doc_idx: int, index of the document in `bow`.
        
        :return: cpt_words: int, total number of words in the document.
                 cpt_d_words : int, number of different words in the document.
        """
        cpt_words, cpt_d_words =  0, len(self.bow[doc_idx])
        for t in self.bow[doc_idx]:
            cpt_words += t[1]
        return cpt_words, cpt_d_words
            

    def estimation(self, em_threshold=1e-5, e_threshold=1e-8, max_iter_em=None, max_iter_var=None):
        """
        Estimation of the alpha and beta parameters of the LDA model
        Uses a EM algorithm which functions the following way :
            - E step by computing best variational parameters
            - M step in order to use these parameters to estimate alpha
            and beta which maximize the expectation of the lower bound
            on the log likelihood

        :param em_threshold: float, if the likelihood varies with a value smaller than this
        the EM algorithm is stopped
        :param e_threshold: float, if the likelihood varies with a value smaller than this
        the E step of the EM algorithm for a document is stopped 
        :param max_iter: int, maximum number of iterations for the EM algorithm to run
        :param max_iter_var: int, maximum number of iterations for the algorithm optimizing
        the variational parameters to run

        :return:
            - Beta : the k x V matrix containing distributions of words inside each topic
            - Phi : the M x N x k matrix containing the latent proportions of each word of
            each document across topics. Phi approximates the latent parameter z, thus for 
            each line of theta we find the relationship of a word of the document with a topic.
            - Gamma : the M x k matrix containing the latent proportions of each topic among
            documents. Gamma is used to approximate the latent variable theta. 
            Each coefficient in gamma thus represents the proportion of a certain topic inside
            the document

            Where : k = nb topics, V = nb of distinct words across corpus, M = number of documents
                    N = maximum amount of distinct words per document
        """
        has_converged = False 
        nb_iter = 0

        #Initialize sufficient statistics and parameters
        for i in range(self.nb_topics):
            for j in range(self.nb_terms):
                #Adding a random term for beta to start from somewhere
                self.topic_word[i][j] += 1 / self.nb_terms + np.random.uniform(0)
                self.topic_tot[i] += self.topic_word[i][j]

        self.optimize_beta_alpha()

        #Initializing variational parameters
        gamma = np.zeros((self.nb_docs, self.nb_topics))
        phi = np.zeros((self.nb_docs, self.doc_max_length, self.nb_topics))

        #EM algorithm
        while (not has_converged and ((max_iter_em is None) or (nb_iter < max_iter_em))):

            print("Iteration:", nb_iter)
            likelihood = 0

            self.topic_tot = np.zeros(self.nb_topics)
            self.topic_word = np.zeros((self.nb_topics, self.nb_terms))

            #Expectation computation part : optimization of the variational parameters
            print("E-Step")

            for d in range(self.nb_docs):
                if (d % 100 == 0):
                    print("E-step through", d, "documents")

                ll = self.inference_doc(d, gamma[d], phi[d], conv=e_threshold, max_iter=max_iter_var)

                likelihood += ll

            #Maximization of the expectation : maximize lower bound of the log likelihood of the variational distribution
            print("M-Step")

            self.optimize_beta_alpha()

            if (nb_iter > 0):
                diff = np.abs((prev_likelihood - likelihood))
                if diff < em_threshold:
                    has_converged = True

            prev_likelihood = likelihood

            print("iteration", nb_iter, likelihood)

            nb_iter += 1

        self.gamma = gamma
        self.phi = phi

        return self.beta, gamma, phi


    def optimize_beta_alpha(self):
        """
        M step of the EM algorithm : use of the variational parameters to optimize alpha and beta
        using maximum likelihood estimation by differentiating the tight lower bound on the log likelihood
        and setting it to zero.

        Update is done using sufficient statistics computed at the E-step of the algorithm
        """
        for i in range(self.nb_topics):
            for j in range(self.nb_terms):
                if self.topic_word[i][j] > 0:
                    self.beta[i][j] = np.log(self.topic_word[i][j]) - np.log(self.topic_tot[i])
                else:
                    #To avoid computing a log of 0
                    #Value is completely arbitrary, we were at loss as what to choose
                    #So we imitated an implementation which chose this value
                    self.beta[i][j] = -100

        #print(np.exp(self.beta))

        #Estimating alpha with Newton-Raphson
        #We have given up on this step as the alpha keeps flunking and going to NaN values
        #self.alpha = self.nr_alpha()

    def inference_doc(self, doc_idx, gamma_doc, phi, conv=10e-8, max_iter=None):
        """
        Variational inference algorithm described in the paper
        Uses gamma and phi, two variational parameters used to approximate
        the distribution of the parameters alpha and beta knowing our data (=corpus)

        :param self: this LDA model
        :param doc_idx: int, index of the document on which to perform the inference
        :param gamma_doc: the gamma variational parameter for the document on which we
        perform the inference. Gamma is used to approximate the latent variable theta. 
        Each coefficient in gamma thus represents the proportion of a certain topic inside
        the document
        :param phi: the phi variational parameter for the document on which we perform the inference.
        Phi approximates the latent parameter z, thus for each line of theta we find the relationship
        of a word of the document with a topic.

        :returns: float, lower bound on the log likelihood for a specific document.
        Updates sufficient statistics and values of phi and gamma as a side effect.
        """
        has_converged = False
        nb_iter = 0
        _, nb_d_words_doc = self.nb_terms_doc(doc_idx)
        phi_prev = np.zeros(self.nb_topics)

        #Initializing variational parameters
        #total count of words inside document
        #and total count of distinct words inside document
        nb_words_doc, nb_d_words_doc = self.nb_terms_doc(doc_idx)
        for n in range(nb_d_words_doc):
            phi[n, :] = 1 / self.nb_topics
        gamma_doc[:] = self.alpha + nb_words_doc / self.nb_topics

        #Storing digamma values to prevent uncessary and redundant digamma computations
        dig_gamma = digamma(gamma_doc)

        #Now estimating gamma and phi
        while (not has_converged and ((max_iter is None) or (nb_iter < max_iter))):

            for i in range(nb_d_words_doc):
                #Manipulating phi in log space in order to prevent numerical underflow
                #Computation of phi normalization constant for this line of phi
                norm_phi = logsumexp(phi[i])

                #Storing previous value of phi important for gamma update
                phi_prev = phi[i]

                #Optimizing gamma using the topic multinomials estimated in beta
                for j in range(self.nb_topics):
                    phi[i][j] = dig_gamma[j] + self.beta[j][self.bow[doc_idx][i][0]]

                #Normalizing phi and switching from log space to probabilities
                phi[i] = np.exp(phi[i] - norm_phi)

                for j in range(self.nb_topics):

                    #Gamma update as specified in the paper
                    gamma_doc[j] += self.bow[doc_idx][i][1] * (phi[i][j] - phi_prev[j])
                
                #Updating values of digamma of gamma
                dig_gamma = digamma(gamma_doc)

            #Computing lower bound on the log likelihood
            likelihood = self.likelihood(doc_idx, nb_d_words_doc, gamma_doc, phi)

            #print("document", doc_idx, "likelihood", likelihood, "nb_iter", nb_iter)

            #Convergence criteria
            if nb_iter > 0:
                diff = np.abs((prev_likelihood - likelihood))
                if diff < conv:
                    has_converged = True
                
            prev_likelihood = likelihood

            nb_iter += 1

        #Updating the sufficient statistics, namely :
        #   - The occurrences of words per topic estimated during the inference phase
        #   this will come in handy during the update of Beta to have directly the counts
        #   of words per topic inside documents
        #   - The total occurrence of words inside a topic, necessary for normalizing multinomials
        #   accross lines of the beta matrix
        for i in range(nb_d_words_doc):
            for j in range(self.nb_topics):
                self.topic_word[j][self.bow[doc_idx][i][0]] += phi[i][j] * self.bow[doc_idx][i][1]
                self.topic_tot[j] += self.bow[doc_idx][i][1] * phi[i][j]

        return likelihood


    def likelihood(self, doc_idx, nb_d_words_doc, gamma_doc, phi):
        """
        Computation of the likelihood lower bound as described in the paper

        :param doc_idx: the document on which the lower bound of the log
        likelihood is computed
        :param nb_d_words_doc: the number of distinct words in the document
        :param gamma_doc: variational parameter gamma for this document
        :param phi: variational parameter phi for this document

        :return: float, the lower bound on the log likelihood for this document
        """
        likelihood = 0.0
        
        #Saving computations with recurrent terms
        dig_gamma = digamma(gamma_doc)
        gamma_sum = np.sum(gamma_doc)
        dig_gamma_sum = digamma(gamma_sum)

        likelihood = loggamma(self.alpha * self.nb_topics) - \
            self.nb_topics * loggamma(self.alpha) - loggamma(gamma_sum)

        
        likelihood += np.sum((self.alpha - 1) * (dig_gamma - dig_gamma_sum) + \
            loggamma(gamma_doc) - (gamma_doc - 1) * (dig_gamma - dig_gamma_sum))

        for i in range(self.nb_topics):
            for j in range(nb_d_words_doc):
                if phi[j][i] > 0:
                    likelihood += self.bow[doc_idx][j][1] * (phi[j][i] * (dig_gamma[i] - dig_gamma_sum) - np.log(phi[j][i]))\
                        + self.beta[i][self.bow[doc_idx][j][0]]

        return likelihood

    
    def display_word_topic_association(self):
        """
        Method used to intepret the learned beta parameter.
        As a reminder, beta is of size nb_topics * nb_terms and
        proba(term | topic) = beta[topic][term]

        We shall for each topic find the top 20 words that contribute 
        to a document being classified as said topic
        """
        top_20_per_topic = np.argsort(self.beta * (-1), axis=1)
        for i in range(self.nb_topics):
            for j in range(self.nb_terms):
                if top_20_per_topic[i][j] < 20:
                    print(self.index[j], end=" ")
            print()



if __name__ == "__main__":
    """
    Example of application using newsgroups
    """
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    pp = Preprocessing()

    index, bow = pp.build_bow(pp.corpus_preproc(train["data"]))

    lda = LDA(5, bow, index, alpha=0.1, set_alpha=True)

    lda.estimation(max_iter_em=100, max_iter_var=10)

    lda.display_word_topic_association()