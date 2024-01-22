package example.ASPIRE.MyoHMI_Android;

import static example.ASPIRE.MyoHMI_Android.ListActivity.getNumChannels;

import android.app.Activity;
import android.util.Log;
import android.widget.Toast;

import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ExecutionException;

import smile.classification.AdaBoost;
import smile.classification.DecisionTree;
import smile.classification.KNN;
import smile.classification.LDA;
import smile.classification.LogisticRegression;
import smile.classification.NeuralNetwork;
import smile.classification.SVM;
import smile.math.Math;
import smile.math.kernel.LinearKernel;


/**
 * Created by Alex on 7/3/2017.
 */

public class Classifier {
    static int numFeatures = 6;
    static double[][] trainVectorP;

    static double[][] train_X;
    static int[] train_Y;
    static double[][] test_X;
    static int[] test_Y;

    int EMG_Window_Size = 32; // How many samples per Myo Channel

    static LDA lda;
    static LDA lda_test;
    static SVM svm;
    static LogisticRegression logit;
    static DecisionTree tree;
    static NeuralNetwork net;
    static KNN knn;
    static AdaBoost forest;
    CNN cnn;
    static int[] classes;
    static Activity activity;
    static int choice = 0; //default CNN, no need to care case 3

    private final int NUM_EPOCHS = 10; // 10

    final int CNN_choice_checker = 0;
    final int LDA_choice_checker = 1;

    static int choice2;

    //classifier trained booleans (just 1 for now to test)
    static boolean trainedLDA;
    static boolean trainedSVM;
    static boolean trainedLOGIT;
    static boolean trainedTREE;
    static boolean trainedNET;
    static boolean trainedKNN;
    static boolean trainedFOREST;
    static boolean trainedCNN;
    static boolean trained2 = false;
    static int nIMUSensors = 0;
    static FeatureCalculator fcalc2 = new FeatureCalculator();
    private static boolean trained = false;
    static private int classSize;
    double[] features;
    int samples = 100;
    double[] mins;
    double[] maxs;
    private String TAG = "Classifier";
    private int prediction;
    int[][][] set;

    // Whether or not the CNN been trained yet
    boolean isCNNTrained = false;

    public Classifier(Activity activity) {
        this.activity = activity;
        this.cnn = new CNN(activity);
    }

    public Classifier() {

    }

    public static void reset() {//reset button from ClassificationFragment
        classes = null;
        trainVectorP = null;
        trained = false;

        trainedLDA = false;
        trainedSVM = false;
        trainedLOGIT = false;
        trainedTREE = false;
        trainedNET = false;
        trainedKNN = false;
        trainedFOREST = false;
        trainedCNN = false;
    }

    public void setnIMUSensors(int imus) {
        nIMUSensors = imus;
    }

    public void Train(ArrayList<DataVector> trainVector, ArrayList<Integer> Classes) {

        classSize = Classes.size();
        classes = new int[classSize];
        int nSensors = getNumChannels();

        trainVectorP = new double[trainVector.size()][numFeatures * nSensors + nIMUSensors]; // nIMUSensors = 0
        for (int i = 0; i < trainVector.size(); i++) {
            for (int j = 0; j < numFeatures * nSensors + nIMUSensors; j++) {
                trainVectorP[i][j] = trainVector.get(i).getValue(j).doubleValue();
            }
        }

        for (int j = 0; j < Classes.size(); j++) {
            classes[j] = Classes.get(j);
        }

        train_X = trainVectorP;
        train_Y = classes;
//
//        // Create a HashMap to store the windows for each class
//        HashMap<Integer, HashMap<Integer, double[]>> hashMap_EMG_Windows = new HashMap<>();
//        // Iterate over the batch labels
//        for (int i = 0; i < classSize; i++) {
//            // Get the current window
//            double[] current_feature = trainVectorP[i];
//            int current_class = classes[i];
//            // Check if the label for the current window is 1.0
//            if (!hashMap_EMG_Windows.containsKey(current_class)) {
//                hashMap_EMG_Windows.put(current_class, new HashMap<Integer, double[]>());
//            }
//            hashMap_EMG_Windows.get(current_class).put(hashMap_EMG_Windows.get(current_class).size(), current_feature);
//        }
//
//        // Find the smallest size in number in the HashMap
//        int smallestSize = Integer.MAX_VALUE;
//        for (Map.Entry<Integer, HashMap<Integer, double[]>> entry : hashMap_EMG_Windows.entrySet()) {
//            if (entry.getValue().size() < smallestSize) {
//                smallestSize = entry.getValue().size();
//            }
//        }
//        int num_gestures = hashMap_EMG_Windows.size();
//
//
//        double split_size = 0.8;
//        int num_train = (int) java.lang.Math.round(smallestSize * num_gestures * split_size);
//        int num_test = (int) ((smallestSize * num_gestures) - num_train);
//
//        train_X = new double[num_train][numFeatures * nSensors + nIMUSensors];
//        train_Y = new int[num_train];
//        test_X = new double[num_test][numFeatures * nSensors + nIMUSensors];
//        test_Y = new int[num_test];
//
//
//        System.out.println("Total amount of training samples: " + num_train);
//        System.out.println("Total amount of testing samples: " + num_test);
//
//
//        for (int i = 0; i < smallestSize; i++) {
//            for (int j = 0; j < num_gestures; j++) {
//                int curr_index = (i*num_gestures)+j;
//
//                if (curr_index < num_train) {
//                    train_X[curr_index] = hashMap_EMG_Windows.get(j).get(i);
//                    train_Y[curr_index] = j;
//                } else {
//                    test_X[curr_index-num_train] = hashMap_EMG_Windows.get(j).get(i);
//                    test_Y[curr_index-num_train] = j;
//                }
//            }
//        }
//
        trained = true;
        trained2 = true;
        switch (choice) {
            case CNN_choice_checker:
                trainCNN();
                break;
            case LDA_choice_checker:
                trainLDA();
                break;
        }
    }

    public void setChoice(int newChoice) {
        trained2 = false;{//must re train if the a new classifier is chosen.. NEED feature that checks if one has already been trained so it doesnt train the same one twice!!!
            if (trained) {
                switch (newChoice) {
                    case CNN_choice_checker:
                        trainCNN();
                        choice = newChoice;
                    case LDA_choice_checker:
                        trainLDA();
                        choice = newChoice;
                        break;
                }
            }
        }
        trained2 = true;
    }

    public void featVector(DataVector Features) {
        features = new double[Features.getLength()];
        for (int i = 0; i < Features.getLength(); i++) {
            features[i] = Features.getValue(i).doubleValue();
        }
    }

    //if flag is turned on (found in newChoice), predict or else return 1000
    public int predict(DataVector Features) {
        featVector(Features);
        if (trained2) {
            switch (choice) {
                case LDA_choice_checker:
//                    Log.d(TAG, "LDA");
                    // ??? Not CNN training?
                    Log.d("Features", Arrays.toString(features));
                    prediction = lda.predict(features);
                    Log.d("Prediction", String.valueOf(prediction));
                    break;
            }
            return prediction;
        }
        return -1;
    }

    public void trainLDA() {
        //if selected gestures is not zero
        if (!trainedLDA) {

//            lda = new LDA(trainVectorP, classes, 0); // Training performed here
//            trainedLDA = true;
            long startTime = System.currentTimeMillis();
            lda = new LDA(train_X, train_Y, 0); // Training performed here
            long duration = System.currentTimeMillis() - startTime;
            System.out.println(
                    "Training LDA took a TOTAL amount of " + duration + " ms to complete."
            );
//            lda_test = new LDA(test_X, test_Y, 0);

            double correct_train_preds = 0;
            int[] prediction_outputs_train = new int[train_X.length];
            for (int i = 0; i < train_X.length; i++) {
                double[] EMG_feature = train_X[i];
                int prediction_train = lda.predict(EMG_feature);
                prediction_outputs_train[i] = prediction_train;
            }

//            double correct_test_preds = 0;
//            int[] prediction_outputs_test = new int[test_X.length];
//            for (int i = 0; i < test_X.length; i++) {
//                double[] EMG_feature_test = test_X[i];
//                int prediction_test = lda_test.predict(EMG_feature_test);
//                prediction_outputs_test[i] = prediction_test;
//            }

            for (int i = 0; i < train_Y.length; i++) {
                if (prediction_outputs_train[i] == train_Y[i]) {
                    correct_train_preds += 1.0;
                }
            }
            double train_accuracy = correct_train_preds / ((double) train_Y.length);
            System.out.println(
                    "LDA's Training Accuracy is: " + train_accuracy + "%."
            );

//            for (int i = 0; i < test_Y.length; i++) {
//                if (prediction_outputs_test[i] == test_Y[i]) {
//                    correct_test_preds += 1.0;
//                }
//            }
//            double test_accuracy = correct_test_preds / ((double) test_Y.length);
//            System.out.println(
//                    "LDA's Testing Accuracy is: " + test_accuracy + "%."
//            );



            trainedLDA = true;
        }
        choice = LDA_choice_checker;
    }

//    public void trainSVM() {
//        if (!trainedSVM) {
//            Toast.makeText(activity, "Training SVM", Toast.LENGTH_SHORT).show();
//            svm = new SVM<>(new LinearKernel(), 10.0, classSize, SVM.Multiclass.ONE_VS_ALL);//classSize + 1
//            svm.learn(trainVectorP, classes);
//            svm.finish();
//            trainedSVM = true;
//        }
//        choice = 1;
//    }
//
//    public void trainLogit() {
//        if (!trainedLOGIT) {
//            Toast.makeText(activity, "Training Logit", Toast.LENGTH_SHORT).show();
//            //Log.d("2", "222" + String.valueOf(trainVectorP.length) + " : " + String.valueOf(classes.length));
//            logit = new LogisticRegression(trainVectorP, classes, 0.0, 1E-5, 5000);
//            trainedLOGIT = true;
//        }
//        Log.d("3", "333");
//        choice = 2;
//    }
//
//    public void trainTree() {
//        if (!trainedTREE) {
//            tree = new DecisionTree(trainVectorP, classes, 350);//in theory, greater the integer: more accurate but slower | lower the integer: less accurate but faster however, i didn't notice a difference
//            trainedTREE = true;
//        }
//    }
//
//    public void trainNet() {
//        if (!trainedNET) {
//            double[][] normalized = Normalize(trainVectorP);
//            net = new NeuralNetwork(NeuralNetwork.ErrorFunction.CROSS_ENTROPY, NeuralNetwork.ActivationFunction.SOFTMAX, trainVectorP[0].length, 150, classSize + 1);
//            net.learn(normalized, classes);
//            net.learn(normalized, classes);
//            net.learn(normalized, classes);
//            trainedNET = true;
//        }
//    }
//
//    public void trainKNN() {
//        Log.d(TAG, "Made it to KNN");
//        if (!trainedKNN) {
//            knn = KNN.learn(trainVectorP, classes, (int) Math.sqrt((double) classSize));
//            trainedKNN = true;
//        }
//    }
//
//    public void trainAdaBoost() {
//        if (!trainedFOREST) {
//            forest = new AdaBoost(trainVectorP, classes, 100, 64);
//            trainedFOREST = true;
//        }
//    }

    /**
     * Fine-tunes the model for 1000 Epochs
     */
    public void trainCNN() {
        // Will train only once
        if (!trainedCNN) {
            // Start training
            cnn.enableTraining((epoch, loss) -> {
//                Log.i("training" + epoch, String.valueOf(loss));
                // End training after 1000 epochs, 20
                if(epoch >= NUM_EPOCHS-1) { // if epoch is 20, make sure it is 20 - 1 = 19
                    cnn.disableTraining();
                }
            });
            trainedCNN = true;
        }
        choice = CNN_choice_checker;
    }

    private double[][] Normalize(double[][] feats) {//to normalize data between interval [0,1]
        int rows = feats.length;//800
        int columns = feats[0].length;//48
        maxs = new double[columns];//feats[0];
        mins = feats[0];
        double[][] normalized = new double[rows][columns];

        for (int i = 1; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (feats[i][j] < mins[j]) {
                    mins[j] = feats[i][j];
                }
                if (feats[i][j] > maxs[j]) {
                    maxs[j] = feats[i][j];
                }
            }
//            System.out.print(mins[0]);
//            System.out.println(" : " + String.valueOf(i));
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                normalized[i][j] = (feats[i][j] - mins[j]) / (maxs[j] - mins[j]);
            }
        }

        return normalized;
    }


    public boolean useRaw() {
        if(choice == CNN_choice_checker){
            System.out.println("choice" + choice);
            System.out.println("cnn choice" + CNN_choice_checker);
            System.out.println("LDA choice" + CNN_choice_checker);
            return true;
        } else {
            return false;
        }
    }

    public int predictRaw(int[][] rawData) {
        if(trained2 && trainedCNN) {
            float[][] cnnFeatures = new float[EMG_Window_Size][8];
            // I can just edit the feature shape
            for(int i = 0; i < rawData.length; i++) {
                for(int j = 0; j < rawData[i].length; j++) {
                    cnnFeatures[i][j] = (float) rawData[i][j];
                }
            }
            TransferLearningModel.Prediction[] predictions = cnn.predict(cnnFeatures);
            float maxConfidence = 0F;
            for(int i = 0; i < predictions.length; i++) {
//                Log.d("Confidence" + predictions[i].getClassName(), predictions[i].getConfidence()+"");
//                Log.d("Confidence" + predictions[i].getClassName(), predictions[i].getConfidence() +Arrays.deepToString(cnnFeatures));
                if(predictions[i].getConfidence() > maxConfidence && Integer.parseInt(predictions[i].getClassName()) <= classSize) {
                    maxConfidence = predictions[i].getConfidence();
                    prediction = Integer.parseInt(predictions[i].getClassName()) - 1;
                }
            }
        }
        return prediction;
    }

    public void addToRaw(int[][] window, int classNum, long timestamp) {
        if(!isCNNTrained) {
            cnn = new CNN(activity);
            isCNNTrained = true;
        }

        float[][] cnnFeatures = new float[EMG_Window_Size][8];
        for(int i = 0; i < window.length; i++) {
            for(int j = 0; j < window[i].length; j++) {
                cnnFeatures[i][j] = (float) window[i][j];
            }
        }

        try {
            cnn.addSample(cnnFeatures, String.valueOf(classNum + 1), timestamp).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}