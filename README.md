# MNIST Neural Network

A simple neural network implementation from scratch for MNIST digit recognition using NumPy.

## Overview

This project implements a 2-layer neural network to classify handwritten digits from the MNIST dataset. The network is built from scratch without using high-level machine learning frameworks like TensorFlow or PyTorch.

## Features

- **From-scratch implementation**: Neural network built using only NumPy
- **2-layer architecture**: Input layer (784 neurons) → Hidden layer (10 neurons) → Output layer (10 neurons)
- **Activation functions**: ReLU for hidden layer, Softmax for output layer
- **Training algorithm**: Gradient descent with backpropagation
- **Data visualization**: Display sample images and predictions
- **Performance evaluation**: Training and test accuracy metrics

## Requirements

```
numpy
matplotlib
kaggle
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install numpy matplotlib kaggle
   ```

2. **Set up Kaggle credentials:**
   
   Create a `.env` file in the project directory:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_key
   ```
   
   Or set environment variables:
   ```bash
   export KAGGLE_USERNAME=your_kaggle_username
   export KAGGLE_KEY=your_kaggle_key
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook mnist_neural_network.ipynb
   ```

## Project Structure

```
mnist-nn/
├── mnist_neural_network.ipynb    # Main Jupyter notebook
├── mnist_data/                   # MNIST dataset files (auto-downloaded)
├── README.md                     # This file
├── .gitignore                   # Git ignore file
└── .env                         # Environment variables (not tracked)
```

## Neural Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (digit classes 0-9)

## Training Process

1. **Data Preprocessing**: 
   - Normalize pixel values to [0, 1]
   - One-hot encode labels
   - Reshape images to vectors

2. **Forward Propagation**:
   - Compute activations through the network
   - Apply ReLU and Softmax activations

3. **Backward Propagation**:
   - Calculate gradients using chain rule
   - Update weights and biases

4. **Training Loop**:
   - 500 iterations with learning rate 0.10
   - Display accuracy every 10 iterations

## Results

The network typically achieves:
- Training accuracy: ~85-90%
- Test accuracy: ~85-88%

## Usage

1. Open `mnist_neural_network.ipynb` in Jupyter Notebook
2. Set your Kaggle credentials in environment variables
3. Run all cells sequentially
4. The notebook will:
   - Download MNIST dataset automatically
   - Train the neural network
   - Display training progress
   - Show sample predictions
   - Evaluate on test set

## Key Functions

- `init_params()`: Initialize network parameters
- `forward_prop()`: Forward propagation
- `backward_prop()`: Backward propagation  
- `gradient_descent()`: Training loop
- `make_predictions()`: Generate predictions
- `get_accuracy()`: Calculate accuracy

## Dataset

The MNIST dataset contains:
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images of handwritten digits (0-9)

Dataset is automatically downloaded from Kaggle using the API.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to fork this project and submit pull requests for improvements.