# Face-Mask-Detection
This is a Face Mask Detection model that uses a convolutional neural network (CNN) to classify images into two categories: "with_mask" and "without_mask". 
![Image about the final project](<Face Mask Detection.png>)

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- NumPy
- OpenCV (cv2)
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow
- Keras

## Overview of the Code
1-Load and preprocess the face mask dataset:
- Load images from the "with_mask" and "without_mask" categories.
- Resize the images to a specified size.
- Shuffle the data.

2-Split the data into input (X) and label (Y) data. Normalize the pixel values.

3-Create a CNN model for image classification:
- Convolutional layers for feature extraction.
- Dense layers for classification.
- Softmax activation for the output layer.

4-Compile and train the model using the training data.

5-Evaluate the model's performance on the test data and display accuracy and loss.

6-Visualize the model's accuracy and loss during training.

7-Create a confusion matrix to evaluate the model's performance.

8-Save the trained model for future use.

9-Create a predictive system that takes an image path as input, preprocesses the image, and uses the model to predict whether the person is wearing a mask ("with_mask" or "without_mask").


## Model Accuracy
The model achieves an accuracy of 95% on the test data.

## Contribution
Contributions to this project are welcome. You can help improve the model's accuracy, explore different CNN architectures, or enhance the data preprocessing and visualization steps. 
Feel free to make any contributions and submit pull requests.
