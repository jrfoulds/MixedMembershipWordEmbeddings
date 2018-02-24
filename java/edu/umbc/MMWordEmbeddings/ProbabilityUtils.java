package edu.umbc.MMWordEmbeddings;

import java.lang.Math;
import java.util.Random;

/** A utility class with static methods for various probability calculations,
 *  including Walker's alias method.
 *  @author James Foulds
*/
public class ProbabilityUtils {
	
	/** Draw a sample from a discrete distribution.
	 * 
	 * @param probs	the probability vector, must sum to one.
	 * @return		the index of the drawn value.
	 */
	public static int sampleFromDiscrete(double[] probs) {
		//Draw a sample from a discrete distribution
		double temp = Math.random();
		double total = 0;
		for (int i = 0; i < probs.length; i++) {
			total += probs[i];
			if (temp < total) {
				return i;
			}
		}
		assert(false);
        return -1;
	}
	
	/** Draw a sample from a discrete distribution, given unnormalized probabilities and their sum.
	 * 
	 * @param unnormalizedProbs		non-negative unnormalized probabilities.
	 * @param sum 					the sum of the unnormalized probabilities (the normalization constant).
	 * @return						the index of the drawn value.
	 */
	public static int sampleFromDiscrete(double[] unnormalizedProbs, double sum) {
		double temp = Math.random() * sum;
		double total = 0;
		for (int i = 0; i < unnormalizedProbs.length; i++) {
			total += unnormalizedProbs[i];
			if (temp < total) {
				return i;
			}
		}
		assert(false);
        return -1;
	}
	
	/** Draw a sample from a discrete distribution, given unnormalized probabilities and their sum.
	 *  This method expects that the probabilities index a sparse subset of the outcomes, with indices
	 *  given in indsForUnnormalizedProbs, and with the numNonSparse non-zero probabilities being the
	 *  first entries into the array.  The remaining entries may be junk, as they are not looked at.
	 * 
	 * @param unnormalizedProbs			non-negative unnormalized probabilities.
	 * @param indsForUnnormalizedProbs	indexes corresponding to entries of unnormalizedProbs. If null, will return the original index
	 * @param sum 						the sum of the unnormalized probabilities (the normalization constant).
	 * @return							the index of the drawn value.
	 */
	public static int sampleFromSparseDiscrete(double[] unnormalizedProbs, int[] indsForUnnormalizedProbs, int numNonSparse, double sum) {
		double temp = Math.random() * sum;
		double total = 0;
		for (int i = 0; i < numNonSparse; i++) {
			total += unnormalizedProbs[i];
			if (temp < total) {
				if (indsForUnnormalizedProbs == null)
					return i;
				else
					return indsForUnnormalizedProbs[i];
			}
		}
		assert(false);
        return -1;
	}

	public static double[] sampleFromDirichlet(double[] dirParams) {
		//Draw a sample from a Dirichlet distribution
		double[] sample = new double[dirParams.length];

		double normConst = 0;
		for (int i = 0; i < sample.length; i++) {
			sample[i] = sampleGamma(dirParams[i], 1);
			normConst += sample[i];
		}
		for (int i = 0; i < sample.length; i++) {
			sample[i] /= normConst;
		}
		//TODO check for NaN
		return sample;
	}
	
	//source: http://vyshemirsky.blogspot.com/2007/11/sample-from-gamma-distribution-in-java.html
	private static Random rng = new Random(java.util.Calendar.getInstance().getTimeInMillis() + Thread.currentThread().getId());
	public static double sampleGamma(double k, double theta) {
		boolean accept = false;
		if (k < 1) {
			 // Weibull algorithm
			 double c = (1 / k);
			 double d = ((1 - k) * Math.pow(k, (k / (1 - k))));
			 double u, v, z, e, x;
			 do {
				  u = rng.nextDouble();
				  v = rng.nextDouble();
				  z = -Math.log(u);
				  e = -Math.log(v);
				  x = Math.pow(z, c);
				  if ((z + e) >= (d + x)) {
					accept = true;
				  }
			 } while (!accept);
			 return (x * theta);
		}
		else {
			 // Cheng's algorithm
			 double b = (k - Math.log(4));
			 double c = (k + Math.sqrt(2 * k - 1));
			 double lam = Math.sqrt(2 * k - 1);
			 double cheng = (1 + Math.log(4.5));
			 double u, v, x, y, z, r;
			 do {
				  u = rng.nextDouble();
				  v = rng.nextDouble();
				  y = ((1 / lam) * Math.log(v / (1 - v)));
				  x = (k * Math.exp(y));
				  z = (u * v * v);
				  r = (b + (c * y) - x);
				  if ((r >= ((4.5 * z) - cheng)) ||	(r >= Math.log(z))) {
					 accept = true;
				  }
			 } while (!accept);
			 return (x * theta);
		}
	}

	/*Given unnormalized log probabilities, compute normalized probabilities in
	 * a numerically stable way.
	 */
	public static void normalizeLogProbs (double[] vals) {
		double max = Double.NEGATIVE_INFINITY;
	    int length = vals.length;
	    double sum = 0;

	    for (int i = 0; i < length; i++) {
	      if (vals[i] > max) {
	        max = vals[i];
	      }
	    }
	    for (int i = 0; i < length; i++) {
		      vals[i] = Math.exp(vals[i] - max);
		      sum += vals[i];
		}
	    for (int i = 0; i < length; i++) {
		      vals[i] = vals[i] / sum;
		}
	}
	
	/*Given unnormalized log probabilities, exp them in
	 * a numerically stable way.  Do not normalize to sum to one,
	 * but return the sum, e.g. for the two parameter version of
	 * sampleFromDiscrete().  The purpose of this method is to
	 * essentially perform the normalizeLogProbs method, but save time
	 * by avoiding the normalization step.
	 */
	public static double expLogProbs (double[] vals) {
		double max = Double.NEGATIVE_INFINITY;
	    int length = vals.length;
	    double sum = 0;

	    for (int i = 0; i < length; i++) {
	      if (vals[i] > max) {
	        max = vals[i];
	      }
	    }
	    for (int i = 0; i < length; i++) {
		      vals[i] = Math.exp(vals[i] - max);
		      sum += vals[i];
		}
	    return sum;
	}
	
	
	public static double expLogProbs (double[] vals, int length) {
		double max = Double.NEGATIVE_INFINITY;
	    double sum = 0;

	    for (int i = 0; i < length; i++) {
	      if (vals[i] > max) {
	        max = vals[i];
	      }
	    }
	    for (int i = 0; i < length; i++) {
		      vals[i] = Math.exp(vals[i] - max);
		      sum += vals[i];
		}
	    return sum;
	}
   //The following code is from the Mallet toolkit
   //McCallum, Andrew Kachites.  "MALLET: A Machine Learning for Language Toolkit."
   //http://mallet.cs.umass.edu. 2002.
   //Used under the terms of the Common Public License, https://opensource.org/licenses/cpl1.0.php
   //According to the code comments, MALLET in turn borrowed this from the Stanford NLP library
   private static final double LOGTOLERANCE = 30.0;
	/**
   * Sums an array of numbers log(x1)...log(xn).  This saves some of
   *  the unnecessary calls to Math.log in the two-argument version.
   * <p>
   * Note that this implementation IGNORES elements of the input
   *  array that are more than LOGTOLERANCE (currently 30.0) less
   *  than the maximum element.
   * <p>
   * Cursory testing makes me wonder if this is actually much faster than
   *  repeated use of the 2-argument version, however -cas.
   * @param vals An array log(x1), log(x2), ..., log(xn)
   * @return log(x1+x2+...+xn)
   */
  public static double sumLogProb (double[] vals)
  {
    double max = Double.NEGATIVE_INFINITY;
    int len = vals.length;
    int maxidx = 0;

    for (int i = 0; i < len; i++) {
      if (vals[i] > max) {
        max = vals[i];
        maxidx = i;
      }
    }

    boolean anyAdded = false;
    double intermediate = 0.0;
    double cutoff = max - LOGTOLERANCE;

    for (int i = 0; i < maxidx; i++) {
      if (vals[i] >= cutoff) {
        anyAdded = true;
        intermediate += Math.exp(vals[i] - max);
      }
    }
    for (int i = maxidx + 1; i < len; i++) {
      if (vals[i] >= cutoff) {
        anyAdded = true;
        intermediate += Math.exp(vals[i] - max);
      }
    }

    if (anyAdded) {
      return max + Math.log(1.0 + intermediate);
    } else {
      return max;
    }

  }
  
  /** Version where the length of a sub-array to consider in the array is specified.
   *  The remainder of the array will be ignored
   * @param vals	log probabilities
   * @param len		number of entries to consider.
   * @return
   */
  public static double sumLogProb (double[] vals, int len)
  {
    double max = Double.NEGATIVE_INFINITY;
    int maxidx = 0;

    for (int i = 0; i < len; i++) {
      if (vals[i] > max) {
        max = vals[i];
        maxidx = i;
      }
    }

    boolean anyAdded = false;
    double intermediate = 0.0;
    double cutoff = max - LOGTOLERANCE;

    for (int i = 0; i < maxidx; i++) {
      if (vals[i] >= cutoff) {
        anyAdded = true;
        intermediate += Math.exp(vals[i] - max);
      }
    }
    for (int i = maxidx + 1; i < len; i++) {
      if (vals[i] >= cutoff) {
        anyAdded = true;
        intermediate += Math.exp(vals[i] - max);
      }
    }

    if (anyAdded) {
      return max + Math.log(1.0 + intermediate);
    } else {
      return max;
    }

  }
  
  /**Build alias table for Walker's method.  This implementation follows the "square histogram"
   * version of Masaglia et al. (2004).
   * @param 	probabilities vector of probabilities that sums to one
   * @parma A 	Pre-allocated probabilities.length x 2 array to store alias table,
   * 			with each row i containing [h, V_i] on return of the method.
   * @param L	Pre-allocated probabilities.length x 2 temp matrix to store low-probability values
   * @param H	Pre-allocated probabilities.length x 2 temp matrix to store high-probability values
   * */
  public static double[][] generateAlias(double[] probabilities) {
	  int l = probabilities.length;
	  double[][] A = new double[l][2];
	  double[][] L = new double[l][2];
	  double[][] H = new double[l][2];
	  generateAlias(probabilities, A, L, H);
	  return A;
  }

  /**Build alias table for Walker's method.  This implementation follows the "square histogram"
   * version of Masaglia et al. (2004). This version is given all pre-allocated matrices to avoid
   * having to re-allocate during the iteration.
   * @param 	probabilities vector of probabilities that sums to one
   * @parma A 	Pre-allocated probabilities.length x 2 array to store alias table,
   * 			with each row i containing [h, V_i] on return of the method.
   * @param L	Pre-allocated probabilities.length x 2 temp matrix to store low-probability values
   * @param H	Pre-allocated probabilities.length x 2 temp matrix to store high-probability values
   * */
  public static void generateAlias(double[] probabilities, double[][] A, double[][] L, double[][] H) {
	  int Lind = 0; //current index into L
	  int Hind = 0; //current index into H
	  int Aind = 0; //current index into A
	  int l = probabilities.length;
	  double lInv = 1.0/l;
	  
	  double updatedHighValue;
	  
	  //create lists of low- and high-probability values
	  //and initialize alias table
	  for (int i = 0; i < l; i++) {
		  if (probabilities[i] <= lInv) {
			  L[Lind][0] = i;
			  L[Lind][1] = probabilities[i];
			  Lind++;
		  }
		  else {
			  H[Hind][0] = i;
			  H[Hind][1] = probabilities[i];
			  Hind++;
		  }
		  
		  A[i][0] = i; //initially point to itself
		  A[i][1] = (i + 1) * lInv;
	  }
	  
	  Lind--; //points to last added entry
	  Hind--;
	  
	  while (Lind >= 0 && Hind >= 0) { //until we remove all low entries from the list
		  //extract next entry from L and H and add map between them to the alias table
		  Aind = (int) L[Lind][0];
		  A[Aind][0] = H[Hind][0]; //h
		  //A[Aind][1] = (Aind - 1) * lInv + L[Lind][1]; 
		  A[Aind][1] = Aind * lInv + L[Lind][1]; //I think the line above needed to be modified vs Masaglia et al. since we're indexing from 0, not 1
		  
		  
		  //add the high-value entry back into the lists with adjusted probability
		  updatedHighValue = H[Hind][1] - (lInv - L[Lind][1]); 
		  Lind--; //we adjust indices here to simplify the indexing above.
		  Hind--; //They now point to the next entry we could get.
		  
		  if (updatedHighValue > lInv) {
			  Hind++;
			  H[Hind][0] = A[Aind][0]; //h
			  H[Hind][1] = updatedHighValue;
		  }
		  else {
			  Lind++;
			  L[Lind][0] = A[Aind][0];
			  L[Lind][1] = updatedHighValue;
		  }
	  }
  }
  
  /** Convert an alias table back into probabilities,
   * for debugging purposes.
   * @param A alias table
   * @return probabilities
   */
  protected static double[] debugAlias(double[][] A) {
	  int l = A.length;
	  double lInv = 1.0/l;
	  double[] probabilities = new double[l];
	  for (int i = 0; i < l; i++) {
		  probabilities[i] += A[i][1] - (lInv * i);
		  probabilities[(int)A[i][0]] += lInv * (i + 1) - A[i][1];
	  }
	  return probabilities;
  }
  
  
  /** Draw a sample from an alias table
   * 
   * @param A alias table
   * @return index of sample into probability vector
   */
  public static int sampleAlias(double[][] A) {
	  int l = A.length;
	  double u = rng.nextDouble();
	  int bin = (int) Math.floor(u * l);
	  if (u < A[bin][1])
		  return bin;
	  else
		  return (int) A[bin][0]; //return alias
  }
  
  /** Draw multiple samples from an alias table and store them in the given array
   * 
   * @param A alias table
   * @param samples		an array to store the samples in
   * @return index of sample into probability vector
   */
  public static void sampleAlias(double[][] A, int[] samples) {
	  int l = A.length;
	  double u;
	  int bin;
	  for (int s = 0; s < samples.length; s++) { //for each sample
		  u = rng.nextDouble();
		  bin = (int) Math.floor(u * l);
		  if (u < A[bin][1])
			  samples[s] = bin;
		  else
			  samples[s] = (int) A[bin][0]; //select the alias value
	  }
  }
  
  
  /* Test Walker's alias method.
   */
  public static void main (String[] args) {
	  double tolerance = 0.00001;
	  double sampleTolerance = 0.001;
	  double[] probabilities;
	  int numSamples = 100000000;
	  int[] samples = new int[numSamples];
	  
	  double[][] testCases = new double[][]{{0.1, 0.2, 0.3, 0.4},
			  								{0.5, 0.5},
			  								{0.3, 0.7},
			  								{0.7, 0.3},
			  								{0.9, 0.05, 0.05},
			  								{},
			  								{},
			  								{},
			  								{},
			  								{}
			  							};
	  for (int j = 0; j < 5; j++) {
		  double[] alpha = new double[(j + 1) * 1000];
		  for (int k = 0; k < alpha.length; k++) 
			  alpha[k] = 0.1;
		  testCases[j + 5] = sampleFromDirichlet(alpha);
	  }
	  
	  
	  for (int j = 0; j < testCases.length; j++) {
		  System.out.println("test case " + j);
		  probabilities = testCases[j];
		  double[][] alias = generateAlias(probabilities);
		  double[] tempProbs = debugAlias(alias);
		  for (int i = 0; i < probabilities.length; i++) {
			  assert(Math.abs(tempProbs[i] - probabilities[i]) < tolerance);
		  }
		  System.out.println("passed distribution asserts!");
		  
		  
		  double[] counts = new double[probabilities.length];
		  for (int k = 0; k < numSamples; k++) {
			  counts[sampleAlias(alias)]++;
		  }
		  for (int k = 0; k < counts.length; k++) {
			  counts[k] = counts[k] / numSamples;
		  }
		  for (int k = 0; k < counts.length; k++) {
			  assert(Math.abs(counts[k] - probabilities[k]) < sampleTolerance);
		  }
		  System.out.println("passed sampling asserts!");
		  
		  
		  sampleAlias(alias, samples);
		  counts = new double[probabilities.length];
		  for (int k = 0; k < numSamples; k++) {
			  counts[samples[k]]++;
		  }
		  for (int k = 0; k < counts.length; k++) {
			  counts[k] = counts[k] / numSamples;
		  }
		  for (int k = 0; k < counts.length; k++) {
			  assert(Math.abs(counts[k] - probabilities[k]) < sampleTolerance);
		  }
		  System.out.println("passed sampling into array asserts!");
		  System.out.println("");
	  }
  }

}
