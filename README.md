# Event-Based(DVS)-Low Precison Autoencoder

Implementation of Low Precision Sparse Autoencoder Model to reduce the computational complexity of Event-based dataset for the inference on Hardware.

# Dependencies
Install Pytorch 1.9.1+cu102, torchsummary, torchaudio and torchvision
Install cv2

# Example

To test the code you can execute "python3 main.py" in your terminal. Before that, you need to provide the dataset path (MNIST-DVS,N-Caltech or N-Cars).
Current test file will execute for MNIST-DVS dataset.
You need to perform pre-processing of data, check "dvs2.tensor.py". For this choose relevant function among load_neuro_events, load_atis_events and load_dvs_events in the same python file for N-Cars, N-Caltech and N-MNIST. 

Finally make relevant changes in get_mnist_dvs function of "mnist_dvs.py" file. 

Now you are all set.
