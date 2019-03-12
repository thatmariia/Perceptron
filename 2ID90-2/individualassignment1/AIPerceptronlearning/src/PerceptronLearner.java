import java.util.List;
import java.util.ArrayList;

/**
 * This class implements Single-layer Perceptron algorithm for n-dimensional
 * vectors.
 *
 * @author Mariia Turchina (1312979)
 */
public class PerceptronLearner {

    /**
     * learning rate of the perceptron
     */
    public final int learningRate = 1; 
    
    /**
     * threshold for positive and negative classification
     */
    public final int threshold = 0;

    /**
     * A function for modifying weights when train item classified incorrectly.
     *
     * @param item train item that has been classified incorrectly
     * @param weight previously used weight that cause incorrect classification
     * @param predicted the result of the classification
     * @param actual the expected classification
     * @return modified weight
     */
    public PVector modifyWeights(PVector item, PVector weight, int predicted, int actual) {
        PVector result = new PVector();

        for (int i = 0; i < item.size(); i++) {
            // using the equation to modify the weights
            result = result.addCoord(weight.getVCoords().get(i) + learningRate * (actual - predicted) * item.getVCoords().get(i));
        }
        return result;
    }

    /**
     * A function for predicting the class of an item by comparing with the
     * threshold.
     *
     * @param item train item that is to be classified
     * @param weights the currently used weights for classification
     * @return predicted classification (true if "+"; false if "-")
     */
    public boolean predict(PVector item, PVector weights) {
        int f = item.dotProduct(weights); // dot product of the item and corresponding weights

        return f > threshold;
    }

    /**
     * A function for training an iteration in the current epoch.
     *
     * @param item train item that is to be classified
     * @param isPositive if the currently considered item should be positively
     * classified
     * @param weights the currently used weights for classification
     * @return weights that has or has not been modified depending on the result
     * of prediction
     */
    public PVector train(PVector item, boolean isPositive, PVector weights) {
        // predicting classification for current item and weights
        boolean predicted = predict(item, weights);
        // Update the weights if prediction is not correct
        if (predicted != isPositive) {
            return modifyWeights(item, weights, predicted ? 1 : 0, isPositive ? 1 : 0);
        }

        return weights;
    }

    /**
     * A function for initializing weight before the first use.
     *
     * @param dimension a dimension of the output weights vector
     * @return vector weights filled with 1's of the specified dimension
     */
    public PVector initializeWeights(int dimension) {
        PVector result = new PVector();

        // adding coordinates 1 to the weight vector
        for (int i = 0; i < dimension; i++) {
            result.addCoord(1);
        }
        return result;
    }

    /**
     * A function that extends vectors with 1's for the biased cases.
     *
     * @param vectors a vector whose dimension is to be extended
     * @return vectors extended with 1's to the next dimension
     */
    public List<PVector> biasVectorChange(List<PVector> vectors) {
        List<PVector> result = new ArrayList<>();

        // adding coordinate 1 to each vector in vectors
        for (int i = 0; i < vectors.size(); i++) {
            PVector v = vectors.get(i).addCoord(1);
            result.add(v);

        }

        return result;
    }

    /**
     * A function that implements perceptron.
     *
     * @param positive vectors that should be positively classified
     * @param negative vectors that should be negatively classified
     * @param bias indication whether bias should be applied to vectors
     * @param maxIterations the max number of epochs available to train
     * perceptron
     * @param queries vectors that should be classified using the perceptron
     * trained with positive and negative
     * @return a string containing the following: - the number of epochs needed
     * to train the perceptron - if trained, the classifications of queries ("+"
     * or "-")
     */
    public String execute(List<PVector> positive, List<PVector> negative, Boolean bias, Integer maxIterations, List<PVector> queries) {
        int dimension = positive.get(0).size();

        // if biased, extending all input vectors with 1's thus extending dimension
        if (bias) {
            dimension++;
            positive = biasVectorChange(positive);
            negative = biasVectorChange(negative);
            queries = biasVectorChange(queries);
        }

        PVector weights = initializeWeights(dimension);

        int epoch_len = positive.size() + negative.size(); // variable for the epoch length

        int correctCounter = 0; // variable for counting correctly classified items
        int epoch;

        // loop through epochs until the max number of iterations achieved or the perceptron is trained
        for (epoch = 0; epoch < maxIterations; epoch++) {
            correctCounter = 0;

            // loop through all positive & negative items to train the perceptron
            for (int epoch_index = 0; epoch_index < epoch_len; epoch_index++) {
                PVector prev_weights = new PVector(weights.getVCoords());

                if (epoch_index < positive.size()) { // training with positively classified vectors
                    int index = epoch_index;
                    weights = train(positive.get(index), true, weights);
                } else {                             // training with negatively classified vectors
                    int index = epoch_index - positive.size();
                    weights = train(negative.get(index), false, weights);
                }
                // if weights have not been changed after training, item has been classified correctly
                if (prev_weights.getVCoords().equals(weights.getVCoords())) {
                    correctCounter++;
                }
            }

            // breaking if the perceptron has been trained
            if (correctCounter >= epoch_len) {
                break;
            }

        }

        // if reached max number of iterations, perceptron hasnt been trained
        // so return the number of epoched needed (max number of iterations)
        if (correctCounter != epoch_len) {
            return String.valueOf(epoch);
        }

        // adding the number of epochs needed to train perceptron to the return string
        String final_result = String.valueOf(epoch + 1) + " ";

        // classifying the query vectors and adding results to the return string
        for (PVector query : queries) {
            final_result += predict(query, weights) ? "+" : "-";
        }

        return final_result;
    }
}
