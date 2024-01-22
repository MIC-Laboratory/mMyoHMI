import config
from dataset import *
import tensorflow as tf
from model import CNN, Model



def convert_and_save(SAVED_MODEL_DIR=config.SAVED_MODEL_DIR):
    emg, label = folder_extract(
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref
    )

    # Extract sEMG signals for wanted gestures.
    gest = gestures(emg, label, targets=config.targets)
    # Perform train test split
    train_gestures, test_gestures = train_test_split(gest)

    # Convert sEMG data to signal images.
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)

    # Convert sEMG data to signal images.
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)

    # tensorflow requires channel last
    X_train = channel_last_transform(X_train)
    X_test = channel_last_transform(X_test)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    X_train = X_train.squeeze()
    X_test = X_test.squeeze()

    print(X_train.shape)
    print(X_test.shape)
    
    train_labels = tf.keras.utils.to_categorical(y_train)
    test_labels = tf.keras.utils.to_categorical(y_test)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=42)
    train_ds = train_ds.batch(config.BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, test_labels))
    test_ds = test_ds.shuffle(buffer_size=len(X_test), seed=42)
    test_ds = test_ds.batch(config.BATCH_SIZE)
    
    
    model = CNN(num_classes=config.num_classes, filters=config.filters, dropout=config.dropout,
            kernel_size=config.k_size, input_shape=config.in_shape, pool_size=config.p_kernel,
            last_block=False)

    new_model = tf.keras.Sequential(model, name="based_model")
    new_model.add(tf.keras.layers.Dense(model.num_classes, name='classifier_block'))
    
    pretraining_model = Model(new_model)
    
    pretraining_model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.on_device_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    )
    
    train_losses = np.zeros([config.NUM_EPOCHS])
    eval_losses = np.zeros([config.NUM_EPOCHS])
    train_accs = np.zeros([config.NUM_EPOCHS])
    eval_accs = np.zeros([config.NUM_EPOCHS])

    for i in range(config.NUM_EPOCHS):
        # training
        for batch_idx, (x,y) in enumerate(train_ds):
            results = pretraining_model.train_PC(x, y)
            train_losses[i] += results["loss"]
            train_accs[i] += results["acc"]

        for batch_idx, (x,y) in enumerate(test_ds):
            results = pretraining_model.on_device_eval(x, y)
            eval_losses[i] += results["loss"]
            eval_accs[i] += results["acc"]
        
        train_losses[i] = train_losses[i] / len(train_ds)
        eval_losses[i] = eval_losses[i] / len(test_ds)
        train_accs[i] = train_accs[i] / len(train_ds)
        eval_accs[i] = eval_accs[i] / len(test_ds)
        
        print(f"Finished {i+1} epoch")
        print(f"  train acc: {train_accs[i]:.3f}" + f"  train loss: {train_losses[i]:.3f}")
        print(f"  eval acc: {eval_accs[i]:.3f}" + f"  eval loss: {eval_losses[i]:.3f}")
    

    tf.saved_model.save(
        pretraining_model,
        SAVED_MODEL_DIR,
        signatures={
            'train':
                pretraining_model.on_device_train.get_concrete_function(),
            'eval':
                pretraining_model.on_device_eval.get_concrete_function(),
            'infer':
                pretraining_model.infer.get_concrete_function(),
        }
    )

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    
    model_file_path = SAVED_MODEL_DIR + '/model.tflite'
    with open(model_file_path, 'wb') as model_file:
        model_file.write(tflite_model)
    
        
if __name__ == "__main__":
    convert_and_save()