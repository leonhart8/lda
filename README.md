# lda
A repo made for working on Latent Dirichlet Analysis

# Installation instructions :

We shall assume that the user already has numpy and scipy installed. In order for all instructions to be launched, especially the preprocessing one needs to install a certain number of packages : 

$ pip3 install nltk
$ pip3 install gensim

# How to use the repository

The repository is composed of two modules :

- preprocessing.py which contains a class that implements the preprocessing pipeline described in the report in details and in the notebook more succintly. It serves one purpose : preparing text data for the LDA model to be trained on it.

- lda.py, a module which contains a LDA class which encapsulates a Latent Dirichlet Allocation model. Documentation is given inside the module and a display of the implementation is given inside the notebook.

One can launch each module separately :

$ python3 lda.py

This launches the module in order to fit over the entire newgroup dataset. Do this only if you have a lot of time on your hands or need to heat your house without turning your radiator on. Very efficient.

$ python3 preprocessing.py

It launches the preprocessing pipeline on the newsgroup dataset and shows a few aspects of it.

Aside from that a notebook is available which gives a few details on the model and provides an illustrated.
The documentation of each function also provides plenty of details on how they are implemented. Docstrings
can be found throughout the modules.

# TO DO

- Implement the newton raphson algorithm for alpha optimization

- Optimize the algorithm further by dropping the for loops using vectoriation and dropping the amount
of calls to digamma

- Better improve the methods to interpret results