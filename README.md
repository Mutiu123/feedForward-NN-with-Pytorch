

## Project Description:
The project involves building a multilayer neural network using Pytorch to classify handwritten digits (0 to 9) from images. Here's a breakdown of the key components:

1. **Dataset**: The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains **28x28 grayscale images** of handwritten digits, each associated with a label (the actual digit it represents).

2. **Model Architecture**: The implemented **multilayer neural network** (also known as a **feedforward neural network**) typically consists of:
    - An **input layer** with **784 neurons** (one for each pixel in the 28x28 image).
    - One **hidden layers** with 500 neurons.
    - An **output layer** with **10 neurons** (one for each digit class).

3. **Activation Functions**:
    - **ReLU (Rectified Linear Unit)** for hidden layers and **softmax** for the output layer (to obtain class probabilities).

4. **Training Process**:
    - PyTorch is used to define the neural network architecture.
    - The training process involves:
        - **Forward pass**: Compute predictions based on input data.
        - **Loss calculation**: Compare predictions with actual labels (using a loss function like **cross-entropy**).
        - **Backpropagation**: Update weights using gradient descent to minimize the loss.
        - **Repeat**: Iterate through the entire dataset multiple times (epochs) to improve model performance.

5. **Hyperparameters**:
    - The hyperparameters include **learning rate**and **batch size** are applied

## Applications:
The digit classification model has practical applications in various domains:
- **Optical Character Recognition (OCR)**: Recognizing handwritten digits in scanned documents.
- **Automated Postal Services**: Sorting mail based on zip codes.
- **Bank Check Processing**: Reading handwritten amounts on checks.
- **Medical Imaging**: Identifying digits in medical images (e.g., patient IDs).

## Methodology:
1. **Data Preprocessing**:
    - Normalize pixel values.
    - Flatten the 28x28 images into a 1D vector.
    - Split the dataset into **training**, **validation**, and **test sets**.

2. **Model Building**:
    - Define the neural network architecture.
    - Initialize weights and biases.
    - The Adam optimizer was applied.

3. **Training**:
    - Feed training data through the network.
    - Compute gradients and update weights.
    - Monitor loss and accuracy on validation set.

4. **Evaluation**:
    - Evaluate model performance on the test set.
    - Calculate the model **accuracy**.

5. **Fine-Tuning**:
    - Experimented the model with different architectures and hyperparameters.


  