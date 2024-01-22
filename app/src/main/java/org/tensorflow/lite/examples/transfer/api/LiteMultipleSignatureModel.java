/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.lite.Interpreter;

import example.ASPIRE.MyoHMI_Android.CNN;

/** A wrapper for TFLite model with multiple signature runner. */
public class LiteMultipleSignatureModel implements Closeable {

  //private static final int BOTTLENECK_SIZE = 1280;
  //private static final int BOTTLENECK_SIZE = 1792;
  private static final int BOTTLENECK_SIZE = 8; // 8
  private static final int EXPECTED_BATCH_SIZE = 280;
  private final Interpreter interpreter;
  private final int numClasses;

  /**
   * Constructor for the multiple signature model wrapper.
   *
   * @param tfLiteModel byte buffer of the saved flatbuffer model
   * @param numClasses number of class labels
   */
  LiteMultipleSignatureModel(ByteBuffer tfLiteModel, int numClasses) {
    this.interpreter = new Interpreter(tfLiteModel);
    this.numClasses = numClasses;
  }


  /**
   * Runs one training step with the given bottleneck batches and labels.
   *
   * @param bottlenecks 3-D float array of bottleneck batches of size (BATCH_SIZE, BOTTLENECK_SIZE)
   * @param labels      2-D float array of label batches of size (BATCH_SIZE, NUM_CLASSES)
   * @return the training loss
   */
  float[] runTraining(float[][][] bottlenecks, float[][] labels) {

    Map<String, Object> inputs = new HashMap<>();
    inputs.put("x", bottlenecks);
    inputs.put("y", labels);

    Map<String, Object> outputs = new HashMap<>();
    FloatBuffer loss = FloatBuffer.allocate(1);
    outputs.put("loss", loss);
    FloatBuffer acc = FloatBuffer.allocate(1);
    outputs.put("acc", acc);


    this.interpreter.runSignature(inputs, outputs, "train");
    float[] values = new float[2];
    values[0] = loss.get(0);
    values[1] = acc.get(0);
    return values;
  }

  float[] runEvaluate(float[][][] bottlenecks, float[][] labels) {

    Map<String, Object> inputs = new HashMap<>();
    inputs.put("x", bottlenecks);
    inputs.put("y", labels);

    Map<String, Object> outputs = new HashMap<>();
    FloatBuffer loss = FloatBuffer.allocate(1);
    outputs.put("loss", loss);
    FloatBuffer acc = FloatBuffer.allocate(1);
    outputs.put("acc", acc);


    this.interpreter.runSignature(inputs, outputs, "eval");
    float[] values = new float[2];
    values[0] = loss.get(0);
    values[1] = acc.get(0);
    return values;
  }

  /**
   * Invokes inference on the given image batches.
   *
   * @param testImage 3-D float array of image of size (IMG_SIZE, IMG_SIZE, 3)
   * @return 1-D float array of softmax output of prediction
   */
  float[] runInference(float[][] testImage) {


    // Run the inference.
    Map<String, Object> inputs = new HashMap<>();
    inputs.put("x", new float[][][] {testImage});

    Map<String, Object> outputs = new HashMap<>();
    float[][] output = new float[1][numClasses];
    outputs.put("output", output);
    this.interpreter.runSignature(inputs, outputs, "infer");
    return output[0];
  }

  int getExpectedBatchSize() {
    return EXPECTED_BATCH_SIZE;
  }

  int getNumBottleneckFeatures() {
    return this.interpreter.getInputTensorFromSignature("bottleneck", "train").shape()[1];
  }

  @Override
  public void close() {
    this.interpreter.close();
  }
}
