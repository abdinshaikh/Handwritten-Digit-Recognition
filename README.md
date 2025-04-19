# Handwritten Digit Recognition Using CNN

## Overview
This project focuses on classifying handwritten digits using Convolutional Neural Networks (CNNs). The model is trained on the well-known MNIST dataset and achieved an impressive accuracy of 99%. The project also includes an interactive web application built using Flask, which allows real-time digit recognition.

## Features
- **Digit Classification**: Classifies handwritten digits (0-9) with high accuracy.
- **Optimized CNN Model**: A Convolutional Neural Network architecture is used to classify digits, achieving 99% accuracy.
- **Data Augmentation**: Applied techniques like rotation, scaling, and shifting to enhance model generalization.
- **Web Application**: An interactive Flask app allows users to draw digits in real-time and receive predictions.

## Technologies Used
- **TensorFlow/Keras**: For building and training the CNN model.
- **Flask**: For creating the interactive web application.
- **Python**: The programming language used for the implementation.
- **MNIST Dataset**: A dataset of handwritten digits used for training and testing the model.

## Model Architecture
The CNN model is designed with the following layers:
1. **Convolutional Layers**: To extract features from images.
2. **Max-Pooling Layers**: For reducing the spatial dimensions of the images.
3. **Fully Connected Layers**: To make final predictions.
4. **Activation Function**: ReLU is used for hidden layers, and softmax is used in the output layer to classify the digits.

### Data Augmentation
To enhance the model’s ability to generalize to new, unseen data, several data augmentation techniques were applied:
- Random rotations
- Zooming and scaling
- Shifting and flipping

## Achievements
- **Accuracy**: Achieved 99% accuracy on the MNIST test dataset.
- **Real-time Prediction**: Built a Flask-based web application for real-time digit recognition from user input.

## How to Run
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abdinshaikh/Handwritten-Digit-Recognition
   
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

### Training the Model
1. Train the CNN model on the MNIST dataset:
   ```bash
   python train_model.py
   
### Running the Web Application
1. Launch the Flask web application:
   ```bash
   python app.py
### Results
- The model achieved 99% accuracy on the MNIST test dataset.

- The Flask application allows users to draw digits and get predictions in real-time.

### Dataset
- The model was trained on the MNIST dataset, which contains 28x28 grayscale images of handwritten digits (0-9).

### Future Improvements
- Fine-tuning the CNN architecture for even better performance.

- Extending the web app to recognize digits with varying image sizes and backgrounds.

- Implementing a larger, more diverse dataset for broader digit recognition.
