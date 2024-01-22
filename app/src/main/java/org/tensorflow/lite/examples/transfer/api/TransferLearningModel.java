/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

// Saving EMG data
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import android.content.Context;


import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** Represents a "partially" trainable model that is based on some other, base model. */
public final class TransferLearningModel implements Closeable {
  private Context context;

  int EMG_Window_Size = 32; // How many samples per Myo Channel

  /**
   * Prediction for a single class produced by the model.
   */
  public static class Prediction {
    private final String className;
    private final float confidence;

    public Prediction(String className, float confidence) {
      this.className = className;
      this.confidence = confidence;
    }

    public String getClassName() {
      return className;
    }

    public float getConfidence() {
      return confidence;
    }
  }

//  private static class TrainingSample {
//    float[] bottleneck;
//    float[] label;
//
//    TrainingSample(float[] bottleneck, float[] label) {
//      this.bottleneck = bottleneck;
//      this.label = label;
//    }
//  }
  public static class TrainingSample {
    public long timestamp;

    float[][] bottleneck;
    float[] label;

      // Mayeb it's here
    TrainingSample(float[][] bottleneck, float[] label, long timestamp) {

      this.bottleneck = bottleneck;
      this.label = label;
      this.timestamp = timestamp;
    }
  }


  /**
   * Consumer interface for training loss.
   */
  public interface LossConsumer {
    void onLoss(int epoch, float loss);
  }

  // Setting this to a higher value allows to calculate bottlenecks for more samples while
  // adding them to the bottleneck collection is blocked by an active training thread.
  private static final int NUM_THREADS =
      Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

  private final Map<String, Integer> classes;
  private final String[] classesByIdx;
  private final Map<String, float[]> oneHotEncodedClass;

  private LiteMultipleSignatureModel model;

  public final List<TrainingSample> trainingSamples = new ArrayList<>();
  // classifier.cnn.model.trainingSamples
  // trainingSamples.get(0).timestamp

  // Where to store training inputs.
  private float[][][] trainingBatchBottlenecks;
  private float[][] trainingBatchLabels;

  // Used to spawn background threads.
  private final ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

  // This lock guarantees that only one thread is performing training and inference at
  // any point in time. It also protects the sample collection from being modified while
  // in use by a training thread.
  private final Lock trainingInferenceLock = new ReentrantLock();

  // This lock guards access to trainable parameters.
  private final ReadWriteLock parameterLock = new ReentrantReadWriteLock();

  // Set to true when [close] has been called.
  private volatile boolean isTerminating = false;

  private int numClasses;

  private boolean USE_SAVED_EMG = false; // false means, EMG data will be stored and saved for next time

  public TransferLearningModel(Context context, ModelLoader modelLoader, Collection<String> classes) {

    try {
      this.model =
          new LiteMultipleSignatureModel(
              modelLoader.loadMappedFile("model.tflite"), classes.size());
      System.out.println(this.model);
      this.context = context;

    } catch (IOException e) {
      throw new RuntimeException("Couldn't read underlying model for TransferLearningModel", e);
    }

    numClasses = classes.size();
    classesByIdx = classes.toArray(new String[0]);
    this.classes = new TreeMap<>();
    oneHotEncodedClass = new HashMap<>();
    for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
      String className = classesByIdx[classIdx];
      this.classes.put(className, classIdx);
      oneHotEncodedClass.put(className, oneHotEncoding(classIdx));
    }
  }

  /**
   * Adds a new sample for training.
   *
   * <p>Sample bottleneck is generated in a background thread, which resolves the returned Future
   * when the bottleneck is added to training samples.
   *
   * @param image image RGB data.
   * @param className ground truth label for image.
   */
  public Future<Void> addSample(float[][] image, String className, long timestamp) {
    checkNotTerminating();

    if (!classes.containsKey(className)) {
      throw new IllegalArgumentException(String.format(
          "Class \"%s\" is not one of the classes recognized by the model", className));
    }

    return executor.submit(
        () -> {
          if (Thread.interrupted()) {
            return null;
          }

          trainingInferenceLock.lockInterruptibly();
          try {
//            float[] bottleneck = model.loadBottleneck(image);
//            // This is for training, replace bottleneck with input
//            trainingSamples.add(new TrainingSample(bottleneck, oneHotEncodedClass.get(className)));


            trainingSamples.add(
                    new TrainingSample(image, oneHotEncodedClass.get(className), timestamp)
            );
          } finally {
            trainingInferenceLock.unlock();
          }

          return null;
        });
  }

  /**
   * Trains the model on the previously added data samples.
   *
   * @param numEpochs number of epochs to train for.
   * @param lossConsumer callback to receive loss values, may be null.
   * @return future that is resolved when training is finished.
   */
  public Future<Void> train(int numEpochs, LossConsumer lossConsumer) {

    checkNotTerminating();
    // trainBatchSize is the expected train size

    int trainBatchSize = getTrainBatchSize();
//    if (trainingSamples.size() < trainBatchSize) {
//      System.out.println("Current Training Sample: " + trainingSamples.size() + "and expected batch size is: " + trainBatchSize);
//
//      throw new RuntimeException(
//          String.format(
//              "Too few samples to start training: need %d, got %d",
//              trainBatchSize, trainingSamples.size()));
//    }

    trainingBatchBottlenecks = new float[trainBatchSize][EMG_Window_Size][8];
    trainingBatchLabels = new float[trainBatchSize][this.classes.size()]; // [80][8]

    // Perform Sliding Window Here

    return executor.submit(
        () -> {
          trainingInferenceLock.lock();
          try {
            epochLoop:
            for (int epoch = 0; epoch < numEpochs; epoch++) {
              long startTime_perEpoch = System.currentTimeMillis();
              float train_totalLoss = 0;
              float test_totalLoss = 0;
              float train_totalacc = 0;
              float test_totalacc = 0;

//              int numBatchesProcessed = 0;
              int trainNumBatchesProcessed = 0;
              int testNumBatchesProcessed = 0;

              // Could calculate accuracy here. Only one iteration though
              for (List<TrainingSample> batch : trainingBatches(trainBatchSize)) {
                if (Thread.interrupted()) {
                  break epochLoop;
                }

                // batch.size() = total dataset size.
                for (int sampleIdx = 0; sampleIdx < batch.size(); sampleIdx++) {
                  TrainingSample sample = batch.get(sampleIdx);
                  trainingBatchBottlenecks[sampleIdx] = sample.bottleneck;
                  trainingBatchLabels[sampleIdx] = sample.label;
                }

//                String save_features_path = "/20_sec_features.ser";
//                String save_labels_path = "/20_sec_labels.ser";

                String save_features_path = "/20_sec_features.ser";
                String save_labels_path = "/20_sec_labels.ser";


                if (USE_SAVED_EMG == false){
                  // Serialization for features
                  try {
                    FileOutputStream bottleneckFileOut = new FileOutputStream(context.getFilesDir() + save_features_path);
                    ObjectOutputStream bottleneckObjectOut = new ObjectOutputStream(bottleneckFileOut);
                    bottleneckObjectOut.writeObject(trainingBatchBottlenecks);
                    bottleneckObjectOut.close();
                    bottleneckFileOut.close();
//                    System.out.println("Bottleneck data saved. @" + context.getFilesDir() + save_features_path);
                  } catch (Exception e) {
                    e.printStackTrace();
                  }
                  // Serialization for labels
                  try {
                    FileOutputStream labelsFileOut = new FileOutputStream(context.getFilesDir() + save_labels_path);
                    ObjectOutputStream labelsObjectOut = new ObjectOutputStream(labelsFileOut);
                    labelsObjectOut.writeObject(trainingBatchLabels);
                    labelsObjectOut.close();
                    labelsFileOut.close();
//                    System.out.println("Labels data saved. @" + context.getFilesDir() + save_labels_path);
                  } catch (Exception e) {
                    e.printStackTrace();
                  }
                } else {
                  try {
                    FileInputStream bottleneckFileIn = new FileInputStream(context.getFilesDir() + save_features_path);
                    ObjectInputStream bottleneckObjectIn = new ObjectInputStream(bottleneckFileIn);
                    // Deserialize the object back to its original type
                    float[][][] loaded_trainingBatchBottlenecks = (float[][][]) bottleneckObjectIn.readObject();
                    bottleneckObjectIn.close();
                    bottleneckFileIn.close();

                    trainingBatchBottlenecks = loaded_trainingBatchBottlenecks;

                    System.out.println("Saved EMG features loaded.");
                  } catch (Exception e) {
                    e.printStackTrace();
                  }

                  try {
                    FileInputStream bottleneckFileIn = new FileInputStream(context.getFilesDir() + save_labels_path);
                    ObjectInputStream labelsObjectIn = new ObjectInputStream(bottleneckFileIn);
                    // Deserialize the object back to its original type
                    float[][] loaded_trainingBatchLabels = (float[][]) labelsObjectIn.readObject();
                    labelsObjectIn.close();
                    bottleneckFileIn.close();

                    trainingBatchLabels = loaded_trainingBatchLabels;

                    System.out.println("Saved labels loaded.");
                  } catch (Exception e) {
                    e.printStackTrace();
                  }
                }


                // Create a HashMap to store the windows for each class
                HashMap<Integer, HashMap<Integer, float[][]>> hashMap_EMG_Windows = new HashMap<>();
                // Iterate over the batch labels
                for (int i = 0; i < trainingBatchLabels.length; i++) {
                  // Get the current window
                  float[][] current_window = trainingBatchBottlenecks[i];
                  // Iterate over the labels for the current window
                  for (int j = 0; j < trainingBatchLabels[i].length; j++) {
                    // Check if the label for the current window is 1.0
                    if (trainingBatchLabels[i][j] == 1.0) {
                      // If it is, add the current window to the list of windows for the corresponding class
                      if (!hashMap_EMG_Windows.containsKey(j)) {
                        hashMap_EMG_Windows.put(j, new HashMap<Integer, float[][]>());
                      }
                      hashMap_EMG_Windows.get(j).put(hashMap_EMG_Windows.get(j).size(), current_window);
                    }
                  }
                }

                // Find the smallest size in number in the HashMap
                int smallestSize = Integer.MAX_VALUE;
                for (Map.Entry<Integer, HashMap<Integer, float[][]>> entry : hashMap_EMG_Windows.entrySet()) {
                  if (entry.getValue().size() < smallestSize) {
                    smallestSize = entry.getValue().size();
                  }
                }

                int num_gestures = hashMap_EMG_Windows.size();

                double split_size = 0.8;
                int num_train = (int) Math.round(smallestSize * num_gestures * split_size);
                int num_test = (int) ((smallestSize * num_gestures) - num_train);

                float[][][] train_X = new float[num_train][EMG_Window_Size][8];
                float[][] train_Y = new float[num_train][this.classes.size()];
                float[][][] test_X = new float[num_test][EMG_Window_Size][8];
                float[][] test_Y = new float[num_test][this.classes.size()];


//                System.out.println("Total amount of training samples: " + num_train);
//                System.out.println("Total amount of testing samples: " + num_test);


                for (int i = 0; i < smallestSize; i++) {
                  for (int j = 0; j < num_gestures; j++) {
                    int curr_index = (i*num_gestures)+j;

                    if (curr_index < num_train) {
                      train_X[curr_index] = hashMap_EMG_Windows.get(j).get(i);
                      train_Y[curr_index] = oneHotEncoding(j);
                    } else {
                      test_X[curr_index-num_train] = hashMap_EMG_Windows.get(j).get(i);
                      test_Y[curr_index-num_train] = oneHotEncoding(j);
                    }
                  }
                }


                double percentage = 0.2; // full 20 seconds
                int EMG_DIVISOR = 4; // Because k-folds is grouping class of 4
                int reduce_EMG_set_count_train = (int) Math.floor((num_train / EMG_DIVISOR) * percentage) * EMG_DIVISOR;
                num_train = reduce_EMG_set_count_train;

                int reduce_EMG_set_count_test = (int) Math.floor((num_test / EMG_DIVISOR) * percentage) * EMG_DIVISOR;
                num_test = reduce_EMG_set_count_test;


                // Perform Batch Size Operation Here - For Training
                int train_BATCH_SIZE = num_train;
                System.out.println("Batch Size: " + train_BATCH_SIZE);


                float[][][] newTrain_X = new float[reduce_EMG_set_count_train][EMG_Window_Size][8];
                // Copy the first `reduce_EMG_set_count` elements from train_X to newTrain_X
                for (int i = 0; i < reduce_EMG_set_count_train; i++) {
                  newTrain_X[i] = train_X[i];
                }
                train_X = newTrain_X;

                float[][] newTrain_Y = new float[reduce_EMG_set_count_train][this.classes.size()];
                // Copy the first `reduce_EMG_set_count` elements from train_Y to newTrain_Y
                for (int i = 0; i < reduce_EMG_set_count_train; i++) {
                  newTrain_Y[i] = train_Y[i];
                }
                train_Y = newTrain_Y;

                float[][][] newTest_X = new float[reduce_EMG_set_count_test][EMG_Window_Size][8];
                // Copy the first `reduce_EMG_set_count` elements from test_X to newTest_X
                for (int i = 0; i < reduce_EMG_set_count_test; i++) {
                  newTest_X[i] = test_X[i];
                }
                test_X = newTest_X;

                float[][] newTest_Y = new float[reduce_EMG_set_count_test][this.classes.size()];
                // Copy the first `reduce_EMG_set_count` elements from test_Y to newTest_Y
                for (int i = 0; i < reduce_EMG_set_count_test; i++) {
                  newTest_Y[i] = test_Y[i];
                }
                test_Y = newTest_Y;


                System.out.println("Total amount of training samples: " + num_train);
                System.out.println("Total amount of testing samples: " + num_test);


                float train_denomin = ((float)(num_train)) / ((float)(train_BATCH_SIZE));
                int batch_idx = 0;
                for(int i = 0; i < num_train; i+=train_BATCH_SIZE) {
                  batch_idx += 1;

                  int end = Math.min(i+train_BATCH_SIZE, num_train);
                  float[][][] trainBatch_X = Arrays.copyOfRange(train_X, i, end);
                  float[][] trainBatch_Y = Arrays.copyOfRange(train_Y, i, end);

                  long startTime = System.currentTimeMillis();
                  float[] train_outputs = this.model.runTraining(trainBatch_X, trainBatch_Y);
                  long duration = System.currentTimeMillis() - startTime;
                  System.out.println(
                          "Epoch: " + epoch + " took " + duration + " ms to complete training backward propagation"
                  );

                  train_totalLoss += train_outputs[0];
                  train_totalacc += train_outputs[1];

                  float epoch_in_decimal;
                  if (batch_idx < train_denomin) {
                    epoch_in_decimal = epoch + (batch_idx/train_denomin);
                  } else {
                    epoch_in_decimal = epoch+1;
                  }

                  trainNumBatchesProcessed += 1;

                  float[] test_outputs = this.model.runEvaluate(test_X, test_Y);

                  System.out.println(
                          "Epoch: " + epoch_in_decimal + " Batch Idx: " + batch_idx +
                          " Train Loss: " + train_outputs[0] + " Train Acc: " + train_outputs[1] +
                          " Test Loss: " + test_outputs[0]+ " Test Acc: " + test_outputs[1]
                  );
                }

                float[] test_outputs = this.model.runEvaluate(test_X, test_Y);
                test_totalLoss += test_outputs[0];
                test_totalacc += test_outputs[1];

                testNumBatchesProcessed += 1;



//                // Set Training Timer Here
//                long startTime = System.currentTimeMillis();
//
//
//                float[] train_outputs = this.model.runTraining(train_X, train_Y);
//
//
//                long endTime = System.currentTimeMillis();
//                long duration = endTime - startTime;
//                System.out.println(
//                        "Epoch: " + epoch + " took " + duration + " ms to complete."
//                );

//                train_totalLoss += train_outputs[0];
//                train_totalacc += train_outputs[1];

//                float[] test_outputs = this.model.runEvaluate(test_X, test_Y);
//                test_totalLoss += test_outputs[0];
//                test_totalacc += test_outputs[1];

//                numBatchesProcessed++;
              }

//              float avgTrainLoss = train_totalLoss / numBatchesProcessed;
//              float avgTrainAcc = train_totalacc / numBatchesProcessed;
//              float avgTestLoss = test_totalLoss / numBatchesProcessed;
//              float avgTestAcc = test_totalacc / numBatchesProcessed;

              float avgTrainLoss = train_totalLoss / trainNumBatchesProcessed;
              float avgTrainAcc = train_totalacc / trainNumBatchesProcessed;
              float avgTestLoss = test_totalLoss / testNumBatchesProcessed;
              float avgTestAcc = test_totalacc / testNumBatchesProcessed;

              System.out.println(
                      "Current Epoch: " + (epoch+1) +
                      " Avg.Train Loss: " + avgTrainLoss + " Avg.Train Acc: " + avgTrainAcc +
                      " Avg.Test Loss: " + avgTestLoss + " Avg.Test Acc: " + avgTestAcc
              );

              if (lossConsumer != null) {
                lossConsumer.onLoss(epoch, avgTrainLoss);
              }

              long duration_perEpoch = System.currentTimeMillis() - startTime_perEpoch;
              System.out.println(
                      "Training Epoch: " + epoch + " took " + duration_perEpoch + " ms to complete ."
              );
            }

            return null;
          } finally {
            trainingInferenceLock.unlock();
          }
        });
  }

  /**
   * Runs model inference on a given image.
   *
   * @param image image RGB data.
   * @return predictions sorted by confidence decreasing. Can be null if model is terminating.
   */
  public Prediction[] predict(float[][] image) {
    checkNotTerminating();
    trainingInferenceLock.lock();

    try {
      if (isTerminating) {
        return null;
      }

      float[] confidences;
      parameterLock.readLock().lock();
      try {
        confidences = this.model.runInference(image);
      } finally {
        parameterLock.readLock().unlock();
      }

      Prediction[] predictions = new Prediction[classes.size()];
      for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
        predictions[classIdx] = new Prediction(classesByIdx[classIdx], confidences[classIdx]);
      }

      Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));
      return predictions;
    } finally {
      trainingInferenceLock.unlock();
    }
  }

  private float[] oneHotEncoding(int classIdx) {
    float[] oneHot = new float[numClasses];
    oneHot[classIdx] = 1;
    return oneHot;
  }

  /** Training model expected batch size. */
  public int getTrainBatchSize() {
//    return trainingSamples.size();

    return Math.max(/* at least one sample needed */ 1, trainingSamples.size());

//    return Math.min(
//        Math.max(/* at least one sample needed */ 1, trainingSamples.size()),
//        model.getExpectedBatchSize()
//    );
  }

  /**
   * Constructs an iterator that iterates over training sample batches.
   *
   * @param trainBatchSize batch size for training.
   * @return iterator over batches.
   */
  private Iterable<List<TrainingSample>> trainingBatches(int trainBatchSize) {
    if (!trainingInferenceLock.tryLock()) {
      throw new RuntimeException("Thread calling trainingBatches() must hold the training lock");
    }
    trainingInferenceLock.unlock();
//
//    Collections.shuffle(trainingSamples); // Perform Random Shuffling
    return () ->
        new Iterator<List<TrainingSample>>() {
          private int nextIndex = 0;

          @Override
          public boolean hasNext() {
            return nextIndex < trainingSamples.size();
          }

          @Override
          public List<TrainingSample> next() {
            int fromIndex = nextIndex;
            int toIndex = nextIndex + trainBatchSize;
            nextIndex = toIndex;
            if (toIndex >= trainingSamples.size()) {
              // To keep batch size consistent, last batc h may include some elements from the
              // next-to-last batch.
              return trainingSamples.subList(
                  trainingSamples.size() - trainBatchSize, trainingSamples.size());
            } else {
              return trainingSamples.subList(fromIndex, toIndex);
            }
          }
        };
  }

  private int numBottleneckFeatures() {
    return model.getNumBottleneckFeatures();
  }

  private void checkNotTerminating() {
    if (isTerminating) {
      throw new IllegalStateException("Cannot operate on terminating model");
    }
  }

  /**
   * Terminates all model operation safely. Will block until current inference request is finished
   * (if any).
   *
   * <p>Calling any other method on this object after [close] is not allowed.
   */
  @Override
  public void close() {
    isTerminating = true;
    executor.shutdownNow();

    // Make sure that all threads doing inference are finished.
    trainingInferenceLock.lock();

    try {
      boolean ok = executor.awaitTermination(5, TimeUnit.SECONDS);
      if (!ok) {
        throw new RuntimeException("Model thread pool failed to terminate");
      }

      this.model.close();
    } catch (InterruptedException e) {
      // no-op
    } finally {
      trainingInferenceLock.unlock();
    }
  }
}
