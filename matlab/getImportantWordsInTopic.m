function [topWords, weights] = getImportantWordsInTopic( topic, dictionary, numToGet)
%gets the words in the topic with the heighest weights, and outputs the
%actual words in them and their associated weights
%topic is the word weights (probabilities or counts)
%dictionary is a cell array of words
%numToGet: how many words to report, e.g. 10
    topWords = cell(numToGet, 1);
    weights = zeros(numToGet, 1);
    [topic inds] = sort(topic, 'descend');
    for i = 1:numToGet
        if topic(i) == 0
            topWords = topWords(1:i-1);
            weights = weights(1:i-1);
            break;
        end
        topWords{i} = dictionary{inds(i)};
        weights(i) = topic(i);
    end
end
