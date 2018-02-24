# MixedMembershipWordEmbeddings
Code implementing the algorithms in: [J. R. Foulds. Mixed Membership Word Embeddings for Computational Social Science. Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS), 2018](http://jfoulds.informationsystems.umbc.edu/papers/2018/Foulds2018MMWE_AISTATS.pdf).


## Prerequisites

* Python
* Tensorflow
* Java

It should work under windows and linux.  I haven't tested it on MacOS but the code is implemented using cross-platform languages, so I expect it would work there as well.

## Data format

The input is a single file, with one line per document, and words represented by zero-based dictionary indices.  See NIPS.txt under the data folder for an example.  This file encodes the [NIPS corpus, due to Sam Roweis](https://cs.nyu.edu/~roweis/data.html).  The dictionary is also provided for NIPS, which allows the final results to be interpreted, but is not used by the algorithms.

## Running the code
The first step is to train the MMSG topic model using annealed Metrolopis-Hastings-Walker collapsed Gibbs sampling.  This is implemented in java.  First compile the java code.  From the root directory of the project:
```bash
cd java
javac edu/umbc/MMWordEmbeddings/*.java
cd ..
```

Then, run the code:
```bash
java -cp java edu.umbc.MMWordEmbeddings.MMSkipGramTopicModel_MHW_mixtureOfExperts filename numTopics numDocuments numWords numIterations contextSize alpha_k beta_w doAnnealing annealingFinalTemperature
```
The last two options default to true, and 0.0001, respectively.  To run this on the NIPS corpus for 2000 topics and 1000 iterations, for example, we can use:
```bash
java -cp java edu.umbc.MMWordEmbeddings.MMSkipGramTopicModel_MHW_mixtureOfExperts data/NIPS.txt 2000 1740 13649 1000 5 0.01 0.001 true
```
After running (it may take a while), this results in three files:

* MMskipGramTopicModel_topicAssignments.txt, in a format similar to the input data, but which contains topic assignments for each word
* MMskipGramTopicModel_wordTopicCountsForTopics.txt, which contains the count matrix for the topics (words by topics).  Add the smoothing hyperparameter and normalize the columns to sum to one to obtain the topics' probability distributions over words.
* MMskipGramTopicModel_wordTopicCountsForWords.txt, which contains the count matrix for the words' distributions over topics (words by topics).  Add the smoothing hyperparameter and normalize the rows to sum to one to obtain the words' probability distributions over topics.

Finally, the embeddings are training via NCE, implemented in python using tensorflow.  Edit python/mixedMembershipSkipGramPreClusteredNCE.py to select the hyperparameters for the algorithm, and its input files (the file encoding the documents, and the corresponding MMskipGramTopicModel_topicAssignments.txt).  Then, run the python script:
```bash
python python/mixedMembershipSkipGramPreClusteredNCE.py
```

This outputs three files:
* MMembeddings.txt, the topic embeddings (topics by dimensions)
* MMnce_biases.txt, the bias terms from inside the softmax (one per word, each on its own line)
* MMnce_weights.txt,  the NCE weight parameters, also known as the output embeddings (words by dimensions)
* MMnormalizedEmbeddings.txt, the topic embeddings, normalized to unit length (topics by dimensions). 

Example scripts which run the above on the NIPS corpus are provided in NIPS_demo.sh (bash) and NIPS_demo.bat (windows).

## Author

* [**James Foulds**](http://jfoulds.informationsystems.umbc.edu/)

## License
Licensed under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

## Acknowledgments

* The python code for training the embeddings was based on [tutorial word embedding code by the authors of TensorFlow](https://www.tensorflow.org/tutorials/word2vec).
