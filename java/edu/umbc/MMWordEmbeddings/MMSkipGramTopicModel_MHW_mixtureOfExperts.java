package edu.umbc.MMWordEmbeddings;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/** Implements the collapsed Gibbs sampler for the mixed membership skip-gram topic model.
 *  This version implements a Metropolis-Hastings-Walker algorithm for leveraging sparsity.
 *  The proposal used here does not compute the expensive portion of the proposal based on
 *  the sparse distribution.  Simulated annealing is also implemented.
 *  
 *  This version uses a uniform mixture over the experts, without weighting by alpha.
 *  The proposal is one expert, chosen from the mixture.
 *  @author James Foulds
 */
public class MMSkipGramTopicModel_MHW_mixtureOfExperts extends MMSkipGramTopicModel {
	
	private static final long serialVersionUID = 6795630840471087536L;
	final int MHWnumCachedSamples;
	int[][] MHWcachedSamples; //[word][sample]
	int[] MHWwordSampleInds; //[word]
	
	//store these in the class to avoid re-allocating them, but their values will not be read
	//outside of the creation of the samples via the Walker algorithm
	private double[] MHWtempDistribution;
	private double[][] MHWaliasTable;
	private double[][] MHW_L;
	private double[][] MHW_H;
	
	double[][] MHWstaleDistributions; //[word][K]
	
	private boolean doAnnealing;
	private double annealingAlpha = 0.99;
	private double annealingFinalTemperature = 1;
	private double annealingLambda = 10;
	private int iterationNumber = 0;
	
	public MMSkipGramTopicModel_MHW_mixtureOfExperts(int numTopics, int numWords, double beta,
			double[] alpha, boolean doAnnealing) {
		super(numTopics, numWords, beta, alpha);
		MHWnumCachedSamples = numTopics;
		MHWtempDistribution = new double[numTopics];
		MHWaliasTable = new double[numTopics][2];
		MHW_L = new double[numTopics][2];
		MHW_H = new double[numTopics][2];
		MHWcachedSamples = new int[numWords][MHWnumCachedSamples];
		this.setDoAnnealing(doAnnealing); 

	}
	
	protected int getNextSample(int word) {
		int returner;
		int sampleInd = MHWwordSampleInds[word];
		if (sampleInd < MHWnumCachedSamples) {
			returner = MHWcachedSamples[word][sampleInd];
			MHWwordSampleInds[word]++;
			return returner;
		}
		else {
			//Draw new samples via Walker's alias method

			//construct new distribution
			double normalizer = 0;
			for (int k = 0; k < numTopics; k++) {
				MHWtempDistribution[k]  = (wordTopicCountsForTopics[word][k] + beta)/(topicCounts[k] + betaSum);
				normalizer += MHWtempDistribution[k];
			}
			for (int k = 0; k < numTopics; k++) {
				MHWtempDistribution[k] /= normalizer;
				
				//save the (soon to be) stale distribution.
				MHWstaleDistributions[word][k] = MHWtempDistribution[k];
			}
			
			
			ProbabilityUtils.generateAlias(MHWtempDistribution, MHWaliasTable, MHW_L, MHW_H); //create alias table
			ProbabilityUtils.sampleAlias(MHWaliasTable, MHWcachedSamples[word]); //get samples

			
			returner = MHWcachedSamples[word][0]; //the first new sample
			MHWwordSampleInds[word] = 1; //the one after it
			return returner;
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

        MHWstaleDistributions = new double[numWords][numTopics];
        
        MHWwordSampleInds = new int[numWords];
        for (int i = 0; i < numWords; i++) {
        	MHWwordSampleInds[i] = MHWnumCachedSamples;
        	getNextSample(i); //creates samples for this word and sets the mixture weight for it
        	MHWwordSampleInds[i] = 0;
        }
    }
	
	protected void updateTopicAssignments(int[][] documents, int contextSize) {
    	//for locality of reference, and to avoid reallocation
    	int oldTopic;
    	int newTopic;
    	int word;
    	int contextSizeForWord;
    	int contextWord;
    	    	
    	int[] contextWordCounts = new int[numWords]; //to adjust counts as we go within the context
    	int ind; //index into mixture weights
    	double pi; //Metropolis-Hastings acceptance probability
    	int accepts = 0;
    	int rejects = 0;
    	int chosenTopicCounts;
    	double tempProposalNumerator; //for proposal portion of acceptance ratio
    	double tempProposalDenominator;
    	double log_A_i_newTopic;
    	double log_A_i_oldTopic;
    	double tempNumerator;
    	double tempDenominator;
    	
    	double temperature = 1; //for simulated annealing
    	iterationNumber++;
    	    	
    	for (int i = 0; i < documents.length; i++) {
	       //sample the topic assignments for each word in the ith document
    	   accepts = 0;
    	   rejects = 0;
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
	           	           	           
	           //Proposal mixture is uniform over the words in the context

	           if ((Math.min(topicAssignments[i].length, n + contextSize + 1) - Math.max(0, n - contextSize)) == 1)
	        	   //With a document of length 1, there are no context words.  Fall back on a uniform random proposal over all words
	        	   ind = random.nextInt(numWords);
	           else {
	        	   //do the proper mixture of experts proposal
	        	   ind = (int)(Math.random() * (Math.min(topicAssignments[i].length, n + contextSize + 1) - Math.max(0, n - contextSize)) + Math.max(0, n - contextSize)); 
		           while (ind == n) {
		        	   ind = (int)(Math.random() * (Math.min(topicAssignments[i].length, n + contextSize + 1) - Math.max(0, n - contextSize)) + Math.max(0, n - contextSize)); 
		           }
		           ind = documents[i][ind];
	           }
	           
	           //Draw from chosen distribution
	           assert ind != -1;

        	   newTopic = getNextSample(ind); //use Walker algorithm sample for the distribution ("expert") corresponding to the sampled context word
        	   chosenTopicCounts = wordTopicCountsForWords[word][newTopic];
	              
	           //Compute Metropolis-Hastings acceptance ratio

	           //contribution from model
	           pi =  Math.log(chosenTopicCounts + alpha[newTopic]) - Math.log(wordTopicCountsForWords[word][oldTopic] + alpha[oldTopic]);
	           
	           log_A_i_newTopic = 0; //for acceptance ratio contribution from model
	           log_A_i_oldTopic = 0;
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               if (k == n) {
	                   continue; //current word is not in the context
	               }
	               contextWord = documents[i][k];
	               tempNumerator = (wordTopicCountsForTopics[contextWord][newTopic] + beta + contextWordCounts[contextWord])/(topicCounts[newTopic] + betaSum);
	               tempDenominator = (wordTopicCountsForTopics[contextWord][oldTopic] + beta + contextWordCounts[contextWord])/(topicCounts[oldTopic] + betaSum);
	               
	               log_A_i_newTopic += Math.log(tempNumerator);
	               log_A_i_oldTopic += Math.log(tempDenominator);
	               	               
	               contextWordCounts[contextWord]++;
	           }
	           pi += log_A_i_newTopic - log_A_i_oldTopic; //contribution from model
	           	
	           if (getDoAnnealing()) {
	        	   //simulated annealing, change temperature
		           temperature = getAnnealingFinalTemperature() + getAnnealingLambda() * Math.pow(getAnnealingAlpha(), iterationNumber);
		           pi /= temperature;
	           }
	           
	           //contribution from proposal
	           
	           //Store the stale distributions, in O(KV) memory.
	           tempProposalNumerator = MHWstaleDistributions[ind][oldTopic];
               tempProposalDenominator = MHWstaleDistributions[ind][newTopic];
               //The memory cost of the above could be avoided by assuming that the counts haven't changed since the samples were drawn from the alias table
	           //I believe Li et al. make this same approximation, since their idea is that the topic counts change slowly.  The code would then be:
	           //tempProposalNumerator = (wordTopicCountsForTopics[ind][oldTopic] + beta)/(topicCounts[oldTopic] + betaSum);
               //tempProposalDenominator = (wordTopicCountsForTopics[ind][newTopic] + beta)/(topicCounts[newTopic] + betaSum);

	           pi += Math.log(tempProposalNumerator) - Math.log(tempProposalDenominator);


	           
	           //Accept or reject proposal
	           pi = Math.min(1, Math.exp(pi));
			   if (random.nextDouble() < pi) {
				   topicAssignments[i][n] = newTopic;
				   accepts = accepts + 1;
			   }
			   else {
				   newTopic = oldTopic;
				   rejects = rejects + 1;
			   }
			   //Zero the contextWordCounts again. In O(|context|) time, not |V|!
	           //After this loop, contextWordCounts will be a vector of zeros
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               contextWord = documents[i][k];
	               contextWordCounts[contextWord] = 0;
	           }
	           
	           //update sufficient statistics cache
	           topicCounts[newTopic] += contextSizeForWord;
	           for (int k = Math.max(0, n - contextSize); k < Math.min(topicAssignments[i].length, n + contextSize + 1); k++) {
	               if (k == n) {
	                   continue; //current word is not in the context
	               }
	               wordTopicCountsForTopics[documents[i][k]][newTopic]++;
	           }   
	           wordTopicCountsForWords[word][newTopic]++;
	       }
    	}
    	System.out.print("Accept rate: " + accepts /(accepts + rejects + 0.0));
    	System.out.println(", temperature: " + temperature);
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
            boolean doAnnealing = true; //default
            if (args.length > 8)
                doAnnealing = Boolean.parseBoolean(args[8]);
            
            double annealingFinalTemperature = 0.0001; //default
            if (args.length > 9)
                annealingFinalTemperature = Double.parseDouble(args[9]);
            
            double[] alpha = new double[numTopics];
            for (int i = 0; i < alpha.length; i++) {
                alpha[i] = alphaTemp;
            }
            
            MMSkipGramTopicModel_MHW_mixtureOfExperts mmsgtm = new MMSkipGramTopicModel_MHW_mixtureOfExperts(numTopics, numWords, beta, alpha, doAnnealing);
            mmsgtm.setAnnealingFinalTemperature(annealingFinalTemperature);
        
            mmsgtm.doMCMC(file, numDocuments, numIterations, contextSize, true);
    	}
        catch (Exception e) {
   			e.printStackTrace();
            System.out.println("\nUsage: filename numTopics numDocuments numWords numIterations contextSize alpha_k beta_w doAnnealing annealingFinalTemperature (last two options default to true, and 0.0001, respectively)");
		}
    }

	double getAnnealingFinalTemperature() {
		return annealingFinalTemperature;
	}

	void setAnnealingFinalTemperature(double annealingFinalTemperature) {
		this.annealingFinalTemperature = annealingFinalTemperature;
	}

	double getAnnealingAlpha() {
		return annealingAlpha;
	}

	void setAnnealingAlpha(double annealingAlpha) {
		this.annealingAlpha = annealingAlpha;
	}

	boolean getDoAnnealing() {
		return doAnnealing;
	}

	void setDoAnnealing(boolean doAnnealing) {
		this.doAnnealing = doAnnealing;
	}

	private double getAnnealingLambda() {
		return annealingLambda;
	}

	private void setAnnealingLambda(double annealingLambda) {
		this.annealingLambda = annealingLambda;
	}

}
