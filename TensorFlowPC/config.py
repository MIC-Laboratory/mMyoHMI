"""
    Description: Configuration Variables and Parameters
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
# How many samples each sEMG image channel contains.
window = 32

# Sliding step (for overlapping)
step = 10

# Kernel size for CNN, must be CHANNELS LAST
k_size = (3, 3)

# Input shape for CNN, must be CHANNELS LAST
in_shape = (window, 8, 1)

# Pool kernel for CNN maxpooling, must be CHANNELS LAST
p_kernel = (2, 1) 

# Exercises with dedicated gestures stored
exercises = ["E2"]

# Path of Ninapro DataBase 5 sEMG dataset.
folder_path = "Ninapro_DB5"

# Ninapro DB5 data collected via 2 Myo armband, controls which armband's 8 sensors to collect
myo_pref = "elbow"

# Class of gestures for training finetune-base model.
targets = [0, 13, 15, 17, 18, 25, 26, 27] # [0, 13, 15, 17], [0] + [i for i in range(13, 30)]


# Number of gestures to detect for finetune-base model.
num_classes = len(targets) # 8

# Number of CNN output filters the model contains.
filters = [32, 64]

# Number of neurons for FFN the model contains.
neurons = None

# Dropout rate.
dropout = 0.5

learning_rate = 0.001

on_device_lr = 0.001

NUM_EPOCHS = 80

BATCH_SIZE = 4096

# Number of gestures to detect for on-device training
CONFIDENCE_SLOTS = 8

SAVED_MODEL_DIR = "saved_model"