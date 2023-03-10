# Event-Based(DVS)-Low Precison Autoencoder

Implementation of Low Precision Sparse Autoencoder Model to reduce the computational complexity of Event-based dataset for the inference on Hardware.

## Dependencies
 * Pytorch 1.9.1+cu102
 * cv2

## Training
 * python3 main.py 
    - Provide the dataset path (MNIST-DVS,N-Caltech or N-Cars).
    - Data preprocessing required using dvs2.tensor.py

