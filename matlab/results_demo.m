%print some results after running a model using the java and python code
loadModel %load output files into matlab
alpha_k = 0.01; %hyperparameters, set these accordingly
beta_w = 0.001;

%load dictionary from a text file
dictionaryFile = 'data/NIPSdict.txt'; %set to your dictionary
fid = fopen(dictionaryFile, 'r');
if fid == -1
    error(['Could not open file: ' dictionaryFile]);
end
dictionary = textscan(fid, '%s', 'Delimiter', '\n'); %get each line
fclose(fid);
dictionary = dictionary{1};

%compute probabilities from counts
topics = mmsg_wtc_topics + alpha_k;
topics = bsxfun(@rdivide, topics, sum(topics));

wordDists = mmsg_wtc_words + beta_w;
wordDists = bsxfun(@rdivide, wordDists, sum(wordDists,2));

%compute topics based on word embeddings for MMSG.  These should be similar to the topics based on the counts, but possibly slightly noisier
topicsMMSG = MMnce_weights * MMembeddings';
topicsMMSG = bsxfun(@plus, topicsMMSG, MMnce_biases);
topicsMMSG = exp(topicsMMSG);
topicsMMSG = bsxfun(@rdivide, topicsMMSG, sum(topicsMMSG));

numToGet = 5;
fprintf(['Printing top words for MMSGTM topics\n'])
topWordsMMSGTM = getImportantWordsInAllTopics(topics, dictionary, numToGet)
%topWordsMMSG = getImportantWordsInAllTopics(topicsMMSG, dictionary, numToGet) %print the top words based on the distributions from the embeddings


%print the top topics for a word
targetWord = 'bayesian' %set me
fprintf(['Printing top topics for word: ' targetWord '\n'])
numTopicsToPrint = 3;
dictReverse = containers.Map(dictionary, 1:length(dictionary));
word = dictReverse(targetWord);
[vals, inds] = sort(wordDists(word,:), 'descend');
for i = 1:numTopicsToPrint
    topic = topics(:,inds(i));
    topWords = getImportantWordsInTopic( topic, dictionary, numToGet) %substitute topicsMMSG if desired
end