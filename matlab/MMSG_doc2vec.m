function docVecs = MMSG_doc2vec( documents, MMembeddings, MMnce_biases, MMnce_weights, MMnormalizedEmbeddings, mmsg_wtc_words, beta, contextLength)
    %Compute a vector representing each document, by summing the topic
    %vectors for each word, averaged over the posterior topic assignments.
    %documents is an array of structs, one for each document, where the field
    %"wordVector" is an array of word indices.  Note, matlab expects one-based
    %indices, so these would need to be shifted versus the indices used for java.
    
    dim = size(MMnce_weights, 2);
    docs = length(documents);
    numWords = size(MMnce_weights, 1);
    numTopics = size(mmsg_wtc_words, 2);
    docVecs = zeros(docs, dim);
    
    topicsMMSG = MMnce_weights * MMembeddings';
    topicsMMSG = bsxfun(@plus, topicsMMSG, MMnce_biases);
    topicsMMSG = exp(topicsMMSG);
    topicsMMSG = bsxfun(@rdivide, topicsMMSG, sum(topicsMMSG));
    
    wordDists = mmsg_wtc_words + beta;
    wordDists = bsxfun(@rdivide, wordDists, sum(wordDists,2));
    
    for i = 1:docs
        i
        docLen = length(documents(i).wordVector);
        for j = 1:docLen;
            word = documents(i).wordVector(j);
            posterior = wordDists(word, :);
            for k = (max(1, j - contextLength):min(j + contextLength, docLen)) %for each word in the context
                if k == j
                    continue; %current word is not in the context
                end
                contextWord = documents(i).wordVector(k);
                posterior = posterior .* topicsMMSG(contextWord, :); %Do I need to do this in log space?
            end
            posterior = posterior ./ sum(posterior);
            for t = 1:numTopics
                docVecs(i,:) = docVecs(i,:) + posterior(t) .* MMnormalizedEmbeddings(t,:);
            end
        end
    end
end

