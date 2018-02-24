#!/bin/bash
cd java
javac edu/umbc/MMWordEmbeddings/*.java
cd ..
java -cp java edu.umbc.MMWordEmbeddings.MMSkipGramTopicModel_MHW_mixtureOfExperts data/NIPS.txt 2000 1740 13649 1000 5 0.01 0.001 true
python python/mixedMembershipSkipGramPreClusteredNCE.py