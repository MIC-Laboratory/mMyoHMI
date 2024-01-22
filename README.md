# mMyoHMI Android App

![System](visuals/System.png?raw=true "System")
* Figure 1: *mMyoHMI* System: On-device Deep learning and Machine Learning
Adaption for EMG Pattern Recognition.

## About
mMyoHMI is an mobile Human-Machine Interface (HMI) system designed for real-time interaction using Electromyography (EMG) signals. It leverages the capabilities of the Myo armband, a wearable device that captures EMG signals from the muscles in the forearm. This system uniquely integrates two models: a Deep Learning (DL) model based on Convolutional Neural Networks (CNN) for robust feature extraction and a Machine Learning (ML) model using Linear Discriminant Analysis (LDA) for efficient on-device adaptation. mMyoHMI is tailored to adapt dynamically to biological heterogeneity that exist between different users, enhancing its usability for EMG pattern recognition. The project aims to push the boundaries of EMG-based interfaces, providing a platform for both research and practical implementations in areas like prosthetics, rehabilitation, and interactive systems.

![GUI](visuals/GUI.png?raw=true "GUI")
* Figure 2: *mMyoHMI* Android-based Graphical User Interface

## GUI
1. Connection Tab: To connect a Myo device, users can tap the Myo icon to initiate a scan for Bluetooth Low Energy (BLE) devices. A list of nearby devices will appear for selection. Once a device is selected, the GUI returns to the EMG tab and establishes a connection.

2. Myo Armband Tab: Displays real-time EMG feature visualization after a successful BLE connection is established with the Myo Armband.

3. Real-time EMG Tab: Allows users to monitor EMG signals in real-time, offering visual feedback to assist in refining gesture or muscle activity execution. It shows raw EMG data from each sensor channel.

4. ML Features Tab: Users can choose which features to employ for pattern recognition and view the real-time values of these features on a radar chart.

5. On-Device Learning Tab: Here, users can engage in on-device learning, select a model for real-time EMG signal processing, and determine the number of gestures for classification.