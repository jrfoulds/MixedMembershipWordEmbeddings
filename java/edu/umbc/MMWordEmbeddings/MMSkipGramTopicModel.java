package edu.umbc.MMWordEmbeddings;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.lang.Math;
import java.util.Random;

import edu.umbc.MMWordEmbeddings.ProbabilityUtils;

/** Implements the collapsed Gibbs sampler for the mixed membership skip-gram topic model.
 *  This version is a naive implementation that does not make use of any sparsity optimizations.
 * @author James Foulds
 *
 */
public class MMSkipGramTopicModel implements Serializable {	

	private static final long serialVersionUID = 1L;
	protected final int numTopics;
	protected final int numWords;
    protected int numDocuments;
	
	protected final double beta;
	protected final double betaSum;
    protected double[] alpha;
    
    protected int[][] wordTopicCountsForWords; //(numWords, numTopics)
    protected int[][] wordTopicCountsForTopics; //zeros(numWords, numTopics);
    protected int[] topicCounts; //numTopics;
    
    protected int[][] topicAssignments; //(document, word)
    
    protected Random random = new Random();
    
    public MMSkipGramTopicModel(int numTopics, int numWords, double beta, double[] alpha) {
    	this.numTopics = numTopics;
    	this.numWords = numWords;
    	this.beta = beta;
    	this.betaSum = beta * numWords;
    	this.alpha = alpha;
    	assert(alpha.length == numTopics);
    }
    
    public void doMCMC(int[][] documents, int numIterations, int contextSize, boolean saveLastSample) {
    	numDocuments = documents.length;
    	initialize(documents, contextSize);
    	
    	for (int i = 0; i < numIterations; i++) {
    		System.out.println("Iteration " + (i + 1));
    		long startTime = System.currentTimeMillis();
    		updateTopicAssignments(documents, contextSize);
    		long finishTime = System.currentTimeMillis();
    		System.out.println((finishTime - startTime)/1000.0 + " seconds");
    	}
    	
    	if (saveLastSample) {
    		try {
				saveToText("MMskipGramTopicModel_");
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    }
    
    public void doMCMC(File file, int numDocuments, int numIterations, int contextSize, boolean saveLastSample) throws IOException {
    	//Load from file
    	int[][] documents = new int[numDocuments][];
    	BufferedReader inputStream = null;
    	
    	int j = 0;
    	try {
            inputStream = new BufferedReader(new FileReader(file));

            String l;
            while ((l = inputStream.readLine()) != null) {
            	if (l.length() == 0) {
            		documents[j] = new int[0];
            		j = j + 1;
            		continue;
            	}
                String[] split = l.split(" ");
                documents[j] = new int[split.length];
                for (int k = 0; k < split.length; k++) {
                	documents[j][k] = Integer.parseInt(split[k]);
                }
                j = j + 1;
            }
        } finally {
            if (inputStream != null) {
                inputStream.close();
            }
        }
    	
    	doMCMC(documents, numIterations, contextSize, saveLastSample);
    }
        
    protected void updateTopicAssignments(int[][] documents, int contextSize) {
    	//for locality of reference, and to avoid reallocation
    	int oldTopic;
    	int newTopic;
    	int word;
    	int contextSizeForWord;
    	int contextWord;
    	double[] probs = new double[numTopics];
    	double sumUnnormalizedProbs;
    	
    	int[] contextWordCounts = new int[numWords]; //to adjust counts as we go within the context

    	for (int i = 0; i < documents.length; i++) {
            if (i % 200 == 0)
                System.out.println("document " + i);
	       //sample the topic assignments for each word in the ith document
	       for (int n = 0; n < documents[i].length; n++) {
	           //remove the current assignment from the sufficient statistics cache
	           oldTopic = topicAssignments[i][n];
	           word = documents[i][n];
	           contextSizeForWord = Math.min(topicAssignments[i].length - 1, n + contextSize) - Math.max(0, n - contextSize); //-1 to remove current word from context, +1 because of last - first, cancel each other out
	           
	           wordTopicCountsForWords[word][oldTopic]--;
	           topicCounts[oldTopic] -= contextSizeForWord;
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               if (k == n) {
	                   continue; //current word is not in the context
	               }
	               wordTopicCountsForTopics[documents[i][k]][oldTopic]--;
	           }

	           //number of times this word has been assigned to each topic                  
	           for (int l = 0; l < numTopics; l++) {
	        	   probs[l] = Math.log(wordTopicCountsForWords[word][l] + alpha[l]); 
	           }
	           
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               if (k == n) {
	                   continue; //current word is not in the context
	               }
	               contextWord = documents[i][k];
	               for (int l = 0; l < numTopics; l++) {
	            	   probs[l] = probs[l] + Math.log(wordTopicCountsForTopics[contextWord][l] + beta + contextWordCounts[contextWord]);
	               }
	               contextWordCounts[contextWord]++;
	           }
	           for (int k = 0; k <= contextSizeForWord-1; k++) {//I believe starting at k=0 corrects an off by one from Carpenter's derivation, since the first word doesn't have any counts added previously
	        	   for (int l = 0; l < numTopics; l++) {
	        		   probs[l] = probs[l] - Math.log(topicCounts[l] + betaSum + k); //normalization term corrected for change in counts after each word
	        	   }
	           }
	           sumUnnormalizedProbs = ProbabilityUtils.expLogProbs(probs);
	           
	           //sample the topic assignment from a discrete distribution
	           newTopic = ProbabilityUtils.sampleFromDiscrete(probs, sumUnnormalizedProbs);
	           topicAssignments[i][n] = newTopic;

	           //update sufficient statistics cache
	           wordTopicCountsForWords[word][newTopic]++;
	           topicCounts[newTopic] += contextSizeForWord;
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               if (k == n) {
	                   continue; //current word is not in the context
	               }
	               wordTopicCountsForTopics[documents[i][k]][newTopic]++;
	           }
	           
	           //Zero the contextWordCounts again between iterations. In O(|context|) time, not |V|!
	           //After this loop, contextWordCounts will be a vector of zeros
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               if (k == n) {
	                   continue; //current word is not in the context
	               }
	               contextWord = documents[i][k];
	               contextWordCounts[contextWord] = 0;
	           }
	       }
    	}
    }
    
    protected void initialize(int[][] documents, int contextSize) {
    	topicAssignments = new int[numDocuments][];
        for (int i = 0; i < numDocuments; i++) {
        	topicAssignments[i] = new int[documents[i].length];
        	//initialize topic assignments uniformly at random
        	for (int j = 0; j < topicAssignments[i].length; j++) {
        		topicAssignments[i][j] = random.nextInt(numTopics);
        	}
        }
        
        //initialize count matrices
        recomputeTextSufficientStatistics(documents, contextSize);
    }

    
    /** Compute the sufficient statistics from scratch. */
    protected void recomputeTextSufficientStatistics(int[][] documents, int contextSize) {
    	
        wordTopicCountsForWords = new int[numWords][numTopics];
        wordTopicCountsForTopics = new int[numWords][numTopics];
        topicCounts = new int[numTopics];

        int word;
        int topic;
        for (int i = 0; i < numDocuments; i++) {
            for (int j = 0; j < documents[i].length; j++) {           
               word = documents[i][j];
               topic = topicAssignments[i][j];
               
               wordTopicCountsForWords[word][topic]++; //number of times this word was assigned to this topic
               topicCounts[topic] = topicCounts[topic] + Math.min(topicAssignments[i].length - 1, j + contextSize) - Math.max(0, j - contextSize); //actual number of words in the context, which varies because of beginning/end of document. last index - first index. -1 because of skipped current word, +1 because of last minus first inclusive, cancel each other out.
               for (int k = Math.max(0, j - contextSize); k < Math.min(topicAssignments[i].length, j + contextSize + 1); k++) { //+1 because of java array indexing from 0 corresponding to a < , not <= like in matlab
                   if (k == j) {
                       continue;
                   }
                   //number of times each context word appeared for this topic
                   wordTopicCountsForTopics[documents[i][k]][topic]++;
               }
            }
        }
    }
    
    public int[][] getWordTopicCountsForWords() {
    	return wordTopicCountsForWords;
    }
    public int[][] getWordTopicCountsForTopics() {
    	return wordTopicCountsForTopics;
    }

    public int[] getTopicCounts() {
    	return topicCounts;
    }
    
    public int[][] getTopicAssignments() {
    	return topicAssignments;
    }
    
    public void saveToText(String baseFilename)  throws IOException {
    	final String topicAssignmentName = "topicAssignments.txt";
    	final String wordTopicCountsForWordsName = "wordTopicCountsForWords.txt";
    	final String wordTopicCountsForTopicsName = "wordTopicCountsForTopics.txt";
    	
    	PrintWriter outputStream = null;
    	try {
            outputStream = new PrintWriter(baseFilename + topicAssignmentName);
        	for (int i = 0; i < topicAssignments.length; i++) {
     	       //sample the topic assignments for each word in the ith document
     	       for (int n = 0; n < topicAssignments[i].length; n++) {
     	    	  outputStream.print(topicAssignments[i][n]);
     	    	  if (n < topicAssignments[i].length) {
     	    		 outputStream.print(" ");
     	    	  }
     	       }
     	      outputStream.println("");
        	}   
        }
        finally {
            if (outputStream != null) {
                outputStream.close();
            }
        }
    	
    	int[][] wordTopicCountsForWords = getWordTopicCountsForWords(); //In the sparse subclass this needs to be converted from the sparse rep.
    	int[][] wordTopicCountsForTopics = getWordTopicCountsForTopics(); //in case this becomes sparse also in a future version
    	
    	outputStream = null;
    	try {
            outputStream = new PrintWriter(baseFilename + wordTopicCountsForWordsName);
        	for (int i = 0; i < wordTopicCountsForWords.length; i++) {
     	       //sample the topic assignments for each word in the ith document
     	       for (int n = 0; n < wordTopicCountsForWords[i].length; n++) {
     	    	  outputStream.print(wordTopicCountsForWords[i][n]);
     	    	  if (n < wordTopicCountsForWords[i].length) {
     	    		 outputStream.print(" ");
     	    	  }
     	       }
     	      outputStream.println("");
        	}
        }
        finally {
            if (outputStream != null) {
                outputStream.close();
            }
        }
    	
    	outputStream = null;
    	try {
            outputStream = new PrintWriter(baseFilename + wordTopicCountsForTopicsName);
        	for (int i = 0; i < wordTopicCountsForTopics.length; i++) {
     	       //sample the topic assignments for each word in the ith document
     	       for (int n = 0; n < wordTopicCountsForTopics[i].length; n++) {
     	    	  outputStream.print(wordTopicCountsForTopics[i][n]);
     	    	  if (n < wordTopicCountsForTopics[i].length) {
     	    		 outputStream.print(" ");
     	    	  }
     	       }
     	      outputStream.println("");
        	}
        }
        finally {
            if (outputStream != null) {
                outputStream.close();
            }
        }
    }
    
    public static void main (String[] args) {
        
        try {
            File file=new File(args[0]);
            if (!file.exists())
               System.out.println("File note found");
           
            
            int numTopics = Integer.parseInt(args[1]);
            int numDocuments = Integer.parseInt(args[2]);
            int numWords = Integer.parseInt(args[3]);
            int numIterations = Integer.parseInt(args[4]);
            int contextSize = Integer.parseInt(args[5]);
            double alphaTemp = Double.parseDouble(args[6]);
            double beta = Double.parseDouble(args[7]);
            
            double[] alpha = new double[numTopics];
            for (int i = 0; i < alpha.length; i++) {
                alpha[i] = alphaTemp;
            }
            
            MMSkipGramTopicModel mmsgtm = new MMSkipGramTopicModel(numTopics, numWords, beta, alpha);
            mmsgtm.doMCMC(file, numDocuments, numIterations, contextSize, true);
    	}
        catch (Exception e) {
   			e.printStackTrace();
            System.out.println("\nUsage: filename numTopics numDocuments numWords numIterations contextSize alpha_k beta_w");
		}
    }
}

