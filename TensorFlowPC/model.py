import tensorflow as tf
from sklearn.model_selection import train_test_split
import config



class Model(tf.Module):
  def __init__(self, model):
    self.model = model
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    )

  # The `train` function takes a batch of input images and labels.
  # By the way, in on-device training, the entire dataset is feed in, not mini batches, so just do
  # Train Test Split withint this function
  @tf.function(input_signature=[
      tf.TensorSpec([None] + [dim for dim in config.in_shape[:-1]], tf.float32), # [None, 32, 8]
      tf.TensorSpec([None, config.num_classes], tf.float32),
  ])
  def on_device_train(self, x, y):
    x = tf.expand_dims(x, axis=-1)
    
    # TODO: Verfiy if computing accuracy adds memory and energy consumption for mobile devices
    # Only add this to compute gradients.
    with tf.GradientTape() as tape:
      """
      logits are the outputs of the last layer of a classification model
      before applying a softmax function to convert them into probabilities
      """
      logits = self.model(x)
      loss = self.model.loss(y, logits)
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.model.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables)
      )
    
    ### Get Accuracy
    predictions = tf.nn.softmax(logits, axis=-1)
    correct_predictions = tf.equal(
      tf.argmax(predictions, axis=-1),
      tf.argmax(y, axis=-1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    result = {
      "loss": loss,
      "acc": accuracy,
    }

    return result
  
  @tf.function(input_signature=[
      tf.TensorSpec([None] + [dim for dim in config.in_shape[:-1]], tf.float32), # [None, 32, 8]
      tf.TensorSpec([None, config.num_classes], tf.float32),
  ])
  def on_device_eval(self, x, y):
    #TODO: Verfiy if computing accuracy adds memory and energy consumption for mobile devices
    """
    logits are the outputs of the last layer of a classification model
    before applying a softmax function to convert them into probabilities
    """
    logits = self.model(x)
    loss = self.model.loss(y, logits)
    
    predictions = tf.nn.softmax(logits, axis=-1)
    correct_predictions = tf.equal(
      tf.argmax(predictions, axis=-1),
      tf.argmax(y, axis=-1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    result = {
      "loss": loss,
      "acc": accuracy
    }
    return result
  
  # The `train` function takes a batch of input images and labels.
  @tf.function(input_signature=[
      tf.TensorSpec([None] + [dim for dim in config.in_shape[:-1]], tf.float32), # [None, 32, 8]
      tf.TensorSpec([None, config.num_classes], tf.float32),
  ])
  def train_PC(self, x, y):
    
    x = tf.expand_dims(x, axis=-1)
    #TODO: Verfiy if computing accuracy adds memory and energy consumption for mobile devices
    # Only add this to compute gradients.
    with tf.GradientTape() as tape:
      """
      logits are the outputs of the last layer of a classification model
      before applying a softmax function to convert them into probabilities
      """
      logits = self.model(x)
      loss = self.model.loss(y, logits)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(
      zip(gradients, self.model.trainable_variables)
    )
      
    predictions = tf.nn.softmax(logits, axis=-1)
    correct_predictions = tf.equal(
      tf.argmax(predictions, axis=-1),
      tf.argmax(y, axis=-1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    result = {
      "loss": loss,
      "acc": accuracy  
    }

    return result


  @tf.function(input_signature=[
      tf.TensorSpec([None] + [dim for dim in config.in_shape[:-1]], tf.float32), # [None, 32, 8]
  ])
  def infer(self, x):
    
    x = tf.expand_dims(x, axis=-1)
    logits = self.model(x)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return {
        "output": probabilities,
        "logits": logits
    }


class CNN(tf.keras.Model):
    def __init__(self, num_classes=config.num_classes, filters=config.filters,
                 dropout=config.dropout, input_shape=config.in_shape,
                 kernel_size=config.k_size, pool_size=config.p_kernel, last_block=True):
        super(CNN, self).__init__()
        
        # last_block=False so you can add sequential layer for modification during transfer learning
        self.last_block = last_block
        self.num_classes = num_classes
        
        # CNN Layer
        self.CNN1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            strides=1,
            kernel_size=kernel_size,
            padding="same",
            input_shape=input_shape,
        )
        
        self.CNN2 = tf.keras.layers.Conv2D(
            filters=filters[1],
            strides=1,
            kernel_size=kernel_size,
            padding="same",
        )

        # Batch Norm Layers
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        
        # PReLU Layers
        self.prelu1 = tf.keras.layers.PReLU()
        self.prelu2 = tf.keras.layers.PReLU()
        
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout)
        
        # Max Pooling Layers
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        
        self.flatten = tf.keras.layers.Flatten()
        
        if self.last_block == True:
            self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.CNN1(inputs)
        x = self.batch_norm1(x)
        x = self.prelu1(x)
        x = self.dropout(x)
        x = self.maxpool1(x)

        x = self.CNN2(x)
        x = self.batch_norm2(x)
        x = self.prelu2(x)
        x = self.dropout(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        
        if self.last_block == True:
            x = self.classifier(x)

        return x
    
if __name__ == "__main__":
    # Create instance of CNN model
    model = CNN(num_classes=config.num_classes, filters=config.filters, dropout=config.dropout,
                kernel_size=config.k_size, input_shape=config.in_shape, pool_size=config.p_kernel,
                last_block=False)
    # Generate random input tensor
    input_tensor = tf.random.normal((1, 32, 8, 1))
    # Get model predictions
    predictions = model(input_tensor)
    # Print predictions
    pred = predictions.numpy()
    # print(pred.shape)