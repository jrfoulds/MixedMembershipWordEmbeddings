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

/** Implements the naive Bayes skip-gram topic model. This model does not require any iterations!
 *  This version is a naive implementation that does not make use of sparsity - the main downside of this is the memory requirements.
 * @author James Foulds
 */
public class NaiveBayesSkipGramTopicModel implements Serializable {	

	private static final long serialVersionUID = 1L;
	protected final int numTopics;
	protected final int numWords;
    protected int numDocuments;
	
	protected final double beta;
	protected final double betaSum;
    
    protected int[][] wordTopicCountsForTopics; //zeros(numWords, numTopics);
    protected int[] topicCounts; //numTopics;
    
    protected Random random = new Random();
    
    public NaiveBayesSkipGramTopicModel( int numWords, double beta) {
    	this.numTopics = numWords;
    	this.numWords = numWords;
    	this.beta = beta;
    	this.betaSum = beta * numWords;
    }
    
    public void doMCMC(int[][] documents, int contextSize, boolean save) {
    	numDocuments = documents.length;
    	recomputeTextSufficientStatistics(documents, contextSize); //This does everything, there are no iterations!
    	
    	if (save) {
    		try {
				saveToText("NaiveBayesSkipGramTopicModel_");
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    }
    
    public void doMCMC(File file, int numDocuments, int contextSize, boolean save) throws IOException {
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
    	
    	doMCMC(documents, contextSize, save);
    }
        
    
    

    
    /** Compute the sufficient statistics from scratch. */
    protected void recomputeTextSufficientStatistics(int[][] documents, int contextSize) {
    	
        wordTopicCountsForTopics = new int[numWords][numTopics];
        topicCounts = new int[numTopics];

        int word;
        int topic;
        for (int i = 0; i < numDocuments; i++) {
            for (int j = 0; j < documents[i].length; j++) {           
               word = documents[i][j];
               topic = word; //This is the crucial point! The word is the cluster assignment, which is observed
               
               topicCounts[topic] = topicCounts[topic] + Math.min(documents[i].length - 1, j + contextSize) - Math.max(0, j - contextSize); //actual number of words in the context, which varies because of beginning/end of document. last index - first index. -1 because of skipped current word, +1 because of last minus first inclusive, cancel each other out.
               for (int k = Math.max(0, j - contextSize); k < Math.min(documents[i].length, j + contextSize + 1); k++) { //+1 because of java array indexing from 0 corresponding to a < , not <= like in matlab
                   if (k == j) {
                       continue;
                   }
                   //number of times each context word appeared for this topic
                   wordTopicCountsForTopics[documents[i][k]][topic]++;
               }
            }
        }
    }
    
    public int[][] getWordTopicCountsForTopics() {
    	return wordTopicCountsForTopics;
    }

    public int[] getTopicCounts() {
    	return topicCounts;
    }
    
    
    public void saveToText(String baseFilename)  throws IOException {
    	final String wordTopicCountsForTopicsName = "wordTopicCountsForTopics.txt";
    	
    	

    	
    	int[][] wordTopicCountsForTopics = getWordTopicCountsForTopics(); //in case this becomes sparse also in a future version
    	
    	
    	PrintWriter outputStream = null;
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
                  
         File file=new File("NIPSfromTextForJava.txt");
         System.out.println(file.exists());
    	
        int numWords = 21101;
        int numDocuments = 1740;
        int contextSize = 10;
        
    	NaiveBayesSkipGramTopicModel nbsgtm = new NaiveBayesSkipGramTopicModel(numWords, 0.001);
    	
    	try {
			nbsgtm.doMCMC(file, numDocuments, contextSize, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
    }

}

