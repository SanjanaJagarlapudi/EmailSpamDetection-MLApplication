
import java.util.*;
import java.io.*; 

/** Email Spam Detection Application
 * Author: Sanjana Jagarlapudi
 * Email: sanjana.jagarlapudi@sjsu.edu
 * 
 */
public class LogisticRegression {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /* Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        public LogisticRegression(int attributes){ //a LogisticRegression Object will take in the number of attributes that are expected 
            weights = new double[attributes];
            for(int i = 0; i < attributes; i++){
               weights[i] = 0; //initialize each weight to 0;
            }
        }

        /**
         * Sigmoid function 
         * This function maps any real value to a value between 0 and 1
         * @param: x - the number to be mapped 
         * @return: double - the sigmoid representation of the input number (a number between 0 and 1)
         * **/
        public double sigmoid(double x){
            return 1 / (1 + Math.exp(-x));
        }

        /** Helper function for prediction
         * This function takes a test instance as input and outputs the probability of the label being 1
         * **/
        /** This function should call sigmoid() **/
        public double predictionHelper(double[] features){
            double sum = 0;
            for(int i = 0; i < features.length; i++){
                sum += weights[i] * features[i]; //applying the corresponing weights to the features
            }
            return sigmoid(sum); //value between 0 and 1
        }


        /* The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call Helper function **/
        public int prediction(double[] features){
            double probability = predictionHelper(features);
            if(probability >= 0.5){
                return 1;
            }
            return 0;
        }

        /** This function takes a test set as input, call the predict function to predict a label for it, 
         * and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix
         * @param: [][]testSet - the 2d array containing the train data, 
         * @param: int[]labels - the array containing the actual labels of each datapoint
         * **/
        public void evaluate(double[][]testSet, int[] labels){
            int truePositives = 0;
            int falsePositives = 0; 
            int trueNegatives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < testSet.length; i++) {
                int predictedLabel = prediction(testSet[i]);
                if (predictedLabel == 1 && labels[i] == 1) { //If the prediced label is true and actual label is true --> true positive value
                    truePositives++;
                } 
                else if (predictedLabel == 1 && labels[i] == 0) { //If the prediced label is true and actual label is false --> false positive value
                    falsePositives++;
                } 
                else if (predictedLabel == 0 && labels[i] == 1) { //If the prediced label is false and actual label is true -->  false negative value
                    falseNegatives++;
                } 
                else {
                    trueNegatives++; //The only other possible case is that it is a true negative
                }
            }

            double positivePrecision = (double) truePositives / (truePositives + falsePositives); // TP/(TP+FP)
            double positiveRecall = (double) truePositives / (truePositives + falseNegatives); // TP/(TP+FN)
            double positiveF1Score = 2 * (positivePrecision * positiveRecall) / (positivePrecision + positiveRecall); // 2rp/(r+p)

            double negativePrecision = (double) trueNegatives / (trueNegatives + falseNegatives); // TN/(TN+FN)
            double negativeRecall = (double) trueNegatives / (trueNegatives + falsePositives);  // TN/(TN+FP)
            double negativeF1Score = 2 * (negativePrecision * negativeRecall) / (negativePrecision + negativeRecall); // 2rp/(r+p)

            double accuracy = (double) (truePositives + trueNegatives) / testSet.length; // TP + TN/(TP+TN+FP+FN)

            //Printing out all the calculated values:
            System.out.println("Accuracy: " + accuracy);
            System.out.println("Positive Class (Spam) Precision: " + positivePrecision);
            System.out.println("Positive Class (Spam) Recall: " + positiveRecall);
            System.out.println("Positive Class (Spam) F1 Score: " + positiveF1Score);
            System.out.println("Negative Class (Ham) Precision: " + negativePrecision);
            System.out.println("Negative Class (Ham) Recall: " + negativeRecall);
            System.out.println("Negative Class (Ham) F1 Score: " + negativeF1Score);
            
            System.out.println("Confusion Matrix:");
            System.out.println("\t\tPredicted");
            System.out.println("\t\tspam\tham");
            System.out.println("Actual\tspam\t" + truePositives + "\t" + falseNegatives);
            System.out.println("\tham\t" + falsePositives + "\t" + trueNegatives);

        
        }

        /** Train the Logistic Regression in a function using Stochastic Gradient Descent 
         * Trains the model and calls evaluate at the end
         * @param trainingSet - The 2D array containing the entire training data set 
         * @param labels - The list of all the resulting labels of the training set
         * 
         * **/
        /** Also compute the log-oss in this function **/
        public void train(double[][] trainingSet, int[] labels){ //the goal is to minimize the loss function
            double totalCost = 0.0; //the variable we will use to keep track of the total cost of our model
            for (int round = 0; round < ITERATIONS; round++) { //for each data point
                double loss = 0.0; //represents the difference between the predicted and actual values
                for (int i = 0; i < trainingSet.length; i++) {
                    double prediction = predictionHelper(trainingSet[i]); 
                    double error = labels[i] - prediction; //error of current data point by comparing probability with actual value
                    for (int j = 0; j < weights.length; j++) {//updating each weight corresponding to each feature
                        weights[j] += rate * error * trainingSet[i][j]; //update the weight variable using the stochastic gradient decent formula
                    }
                }
                //We want to calculate the loss AFTER the weights have already been updated. Thus, lets make another loop, after the weights, to calculate loss
                for(int k = 0; k < trainingSet.length; k++){
                    double prediction = predictionHelper(trainingSet[k]); //get the integer representation of this data point's probability
                    loss += -labels[k] * Math.log(prediction) - (1 - labels[k]) * Math.log(1 - prediction); //update the loss variable using the Log Odds formula
                }
                loss = loss / trainingSet.length; //to get the average loss
                totalCost += loss; //this returns the total cost, which is the average of all the log loss values.
                System.out.println("Current Iteration Number: " + (round + 1) + ", Log Loss: " + loss); //print out the results
            }
            System.out.println("Total cost: " + totalCost/200);//Divide the total cost by 200, to get the average, and print it out. 
            evaluate(trainingSet, labels); //To get other statistics of the training process, call the evaluate method
            
        }

        /** Function to read the input dataset 
         * In this function, we want to read in the file and represent it as a 2D array
         * @param file - the file containing the training data
         * **/
        public void readTrainFile(File file){
            System.out.println("\nTrain Data Statistics: ");
            try{
                BufferedReader br = new BufferedReader(new FileReader(file)); //prepares the file to be read line by line
                //Since we don't initially know how many data points are in the training set, we should use the ArrayList data type.
                ArrayList<double[]> trainingSet = new ArrayList<double[]>(); //The 2D array that will store the training data
                ArrayList<Integer> labelList = new ArrayList<Integer>();//list the label each data point was given, 0 for not spam, 1 for spam
                String line = br.readLine();
                line = br.readLine();//ignore the first line, since its just the attribute names 
                while(line != null){//each subsequent line is one data point, and it contains a list of all the attribute values of that point 
                    String[] elements = line.split(","); //each attribute is seperated by a comma
                    double[] features = new double[elements.length - 1]; //we want to exclude the label (last column)
                    for(int i = 0; i < elements.length - 1; i++){//initializing features for current data point
                        features[i] = (Double.parseDouble(elements[i])); 
                    }
                    int label = Integer.parseInt(elements[elements.length - 1]);
                    trainingSet.add(features); //Add each of the datapoints to the list of all the data points, ie the training set 
                    labelList.add(label);
                    line = br.readLine();
                }
                //Let's convert the ArrayLists to Arrays so they will be easier to handle in the train method and we can work with primitive types 
                double[][] finalTrainingSet = trainingSet.toArray(new double[trainingSet.size()][]);
                int[] finalLabels = labelList.stream().mapToInt(Integer::intValue).toArray(); //Since arraylists use wrapper class types, convert to primitive types
                //call the train method on the 2D array and the labels 
                train(finalTrainingSet, finalLabels); //now that we have the data is a suitable form, train it.
                br.close();
            }
            catch(IOException x){
                System.out.println(x.getMessage());
            }
            
        }

        /** Function to read the test dataset
         * This function is similar to the readTrainFile method, but as we do not want to re-train our model with the test data,
         * it calls the evaluate method instead of the train method
         *  @param: file - the file containing the testing data
         * **/
        public void readTestFile(File file) {
            System.out.println("\nTest Data Statistics: ");
            try {
                BufferedReader br = new BufferedReader(new FileReader(file));
                //since we don't know how many data points in set, use arralist data type, but we'll keep the inner data type as arrays so we can work with the primitive double type
                ArrayList<double[]> testSet = new ArrayList<>();  
                ArrayList<Integer> labelList = new ArrayList<>(); //again, since we don't know how many labels there are YET, --> use arraylist dt
                String line = br.readLine(); // Skip header line since it is just attribute names 
                while ((line = br.readLine()) != null) { //while there is still another data type
                    String[] elements = line.split(","); //since each attribute value is divided by commas --> split by commas
                    double[] features = new double[elements.length - 1]; //this will store the individual attribute values of each data point
                    for (int i = 0; i < elements.length - 1; i++) {
                        features[i] = Double.parseDouble(elements[i]); //initilize features
                    }
                    int label = Integer.parseInt(elements[elements.length - 1]); //since the label is the last column, take the last index of element for each data point
                    testSet.add(features);
                    labelList.add(label);
                }
                double[][] finalTestSet = testSet.toArray(new double[testSet.size()][]);
                int[] finalLabels = labelList.stream().mapToInt(Integer::intValue).toArray();
                evaluate(finalTestSet, finalLabels);
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /** main Function **/
        public static void main(String[] args){
            //Instructions to run the code:
            //Since the paths that are provided are absolute paths local to my computer, replace them with relative paths or the corresponding absolute paths
            //that are relavent to the user's computer
            //After this, you should be able to directly run the code and the outputs will pop up in the output/terminal tab
            File trainFile = new File ("/Users/sanjana/Desktop/CS171MachineLearningProjects/ProgrammingAssignment1/train-1.csv");
            File testfile = new File("/Users/sanjana/Desktop/CS171MachineLearningProjects/ProgrammingAssignment1/test-1.csv");
            //We need to first find how many attributes are in the data set to initilize a LogisticRegression object.
            int size = 0;
            try{
                BufferedReader br = new BufferedReader(new FileReader(trainFile));
                String line = br.readLine();
                String[] attributes = line.split(",");
                size = attributes.length;
                br.close();
            }
            catch (Exception x){
                System.out.println(x.getMessage());
            }
            LogisticRegression lr = new LogisticRegression(size - 1); //-1 because we don't want to include labels as an attribute
            //Call the appropriate methods on the appropriate files
            lr.readTrainFile(trainFile); 
            lr.readTestFile(testfile);
        }
        
    }

