%V is a words x topics matrix of probabilities. Unnormalized counts will
%work for the first argument.
function [topWords, topicWordProbs] = getImportantWordsInAllTopics(V, dictionary, numToGet)
    numTopics = size(V,2);
    topWords = cell(numToGet, numTopics);
    topicWordProbs = zeros(numToGet, numTopics);
    for i = 1:numTopics
        [temp1, temp2] = getImportantWordsInTopic(V(:,i), dictionary, numToGet);
        topWords(1:length(temp1),i) = temp1;
        topicWordProbs(1:length(temp2),i) = temp2;
    end
end