# Aspect-based Sentiment Analysis with Graph Convolution over Syntactic Dependencies
PyTorch implementation of aspect-based sentiment analysis with graph convolution over dependency parse trees of health and well-being related content.

## Repository
* `data` folder contains preprocessed training, validation and test dataset.
*  `glove_dictionary.py` download GloVe model and execute this script in order to create a dictionary.
* `model.py` contains the implementation of the model.
* `main.py` is the script that contains training, validation and testing of the model.

## References
* [Gräßer F, Kallumadi S, Malberg H, Zaunseder S. Aspect-based sentiment analysis of drug reviews applying cross-domain and cross-data learning. In Proceedings of the 2018 International Conference on Digital Health 2018 Apr 23 (pp. 121-125).](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29)
* [Pennington J, Socher R, Manning CD. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) 2014 Oct (pp. 1532-1543).](https://www.aclweb.org/anthology/D14-1162.pdf)
* [Hamilton W.L, Ying R, Leskovec J. Inductive Representation Learning on Large Graphs. NIPS, 2017.](https://papers.nips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)
* [Early Stopping for PyTorch](https://github.com/Bjarten/early-stopping-pytorch)
* [Chen D, Manning CD. A fast and accurate dependency parser using neural networks. InProceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) 2014 Oct (pp. 740-750).](https://www.aclweb.org/anthology/D14-1082.pdf)
