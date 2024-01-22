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

package example.ASPIRE.MyoHMI_Android;

import android.content.Context;
import android.os.ConditionVariable;
import android.util.Log;

import java.io.Closeable;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.transfer.api.ModelLoader;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;

// Added 2 lines below
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.flex.FlexDelegate;


/**
 * App-layer wrapper for {@link TransferLearningModel}.
 *
 * <p>This wrapper allows to run training continuously, using start/stop API, in contrast to
 * run-once API of {@link TransferLearningModel}.
 */
public class CNN implements Closeable {

    // Number of epochs to finetune model for
    private final int NUM_EPOCHS = 10; // 10

    public final TransferLearningModel model;

    private final ConditionVariable shouldTrain = new ConditionVariable();
    private volatile LossConsumer lossConsumer;

    CNN(Context context) {
        // Initialize Deep Learning Model and specify classes
        model =
                new TransferLearningModel(
                        context,
                        new ModelLoader(context, "model"),
                        Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8")
                );


        new Thread(() -> {
            while (!Thread.interrupted()) {
                shouldTrain.block();
                try {
                    long startTime = System.currentTimeMillis();
                    model.train(NUM_EPOCHS, lossConsumer).get();
                    long duration = System.currentTimeMillis() - startTime;
                    System.out.println(
                            "Training took a TOTAL amount of " + duration + " ms to complete."
                      );
                } catch (ExecutionException e) {
                    throw new RuntimeException("Exception occurred during model training", e.getCause());
                } catch (InterruptedException e) {
                    // no-op
                }
            }
        }).start();
    }

    // This method is thread-safe.
    public Future<Void> addSample(float[][] image, String className, long timestamp) {
        return model.addSample(image, className, timestamp);
    }

    // This method is thread-safe, but blocking.
    public Prediction[] predict(float[][] image) {
        return model.predict(image);
    }



    public int getTrainBatchSize() {
        return model.getTrainBatchSize();
    }

    /**
     * Start training the model continuously until {@link #disableTraining() disableTraining} is
     * called.
     *
     * @param lossConsumer callback that the loss values will be passed to.
     */
    public void enableTraining(LossConsumer lossConsumer) {
        this.lossConsumer = lossConsumer;
        shouldTrain.open();
    }

    /**
     * Stops training the model.
     */
    public void disableTraining() {
        shouldTrain.close();
    }

    /** Frees all model resources and shuts down all background threads. */
    public void close() {
        model.close();
    }
}
