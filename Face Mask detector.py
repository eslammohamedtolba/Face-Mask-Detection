# import kaggle module
!pip install kaggle
# configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
# API to fetch the dataset from Kaggle
!kaggle datasets download -d omkargurav/face-mask-dataset
# unzip the zip folder of images
! unzip '/content/face-mask-dataset.zip'


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import cv2
from sklearn.model_selection import train_test_split
# import required dependencies for model creation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense
# import confusion matrix
from tensorflow.math import confusion_matrix
import pickle as pk # save the model


directory = '/kaggle/input/face-mask-dataset/data'
categories = ['with_mask','without_mask']


Image_size = 100
data = []
for category in categories:
    label = categories.index(category)
    folder = os.path.join(directory,category)
    for img_name in os.listdir(folder):
        image_path = os.path.join(folder,img_name)
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image,(Image_size,Image_size))
            data.append([resized_image,label])
        else:
            print(f"Error on the image of path {image_path}")

# Shuffle data 
random.shuffle(data)


# Split data into input and label data
X = []
Y = []
for feature, label in data:
    X.append(feature)
    Y.append(label)
# Show the size of input and label data
print(len(X))
print(len(Y))
# Convert input and label data into numpy array
X = np.array(X)
Y = np.array(Y)
# Show the input and label data size
print(X.shape)
print(Y.shape)
# Show the first image before scaling 
print(X[0])
# Scaling images values
X = X/255
# Show the first image after scaling 
print(X[0])
# Show the first five images and its labels
for i in range(15):
    plt.imshow(X[i])
    plt.show()
    print(Y[i])



# Split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.85,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Create the model
Input_size = (Image_size,Image_size,3) # Determine the input size
# Determine the number of classes
Num_classes = 2

Model = Sequential([
    Input(shape=Input_size),
    Conv2D(32,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(Num_classes,activation='softmax')
])
# Compile the model
Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# train the model
result = Model.fit(x_train,y_train,validation_split=0.1,epochs=7)


# evaluate the model
evaluation = Model.evaluate(x_test,y_test)
print("the loss value is: ",evaluation[0])
print("the accuracy value is: ",evaluation[1])

# Visualize the accuracy with the validation accuracy
plt.figure(figsize=(7,7))
plt.plot(result.history['accuracy'],color='red')
plt.plot(result.history['val_accuracy'],color='blue')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','val_accuracy'],loc='lower right')
# Visualize the loss with the validation loss
plt.figure(figsize=(7,7))
plt.plot(result.history['loss'],color='red')
plt.plot(result.history['val_loss'],color='blue')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['loss','val_loss'],loc='upper right')



# Make the model predict on test input data
predicted_y = Model.predict(x_test)
y_predicted_values = []
for value in predicted_y:
    y_predicted_values.append(np.argmax(value))
comparison = []
for predicted_value,true_value in zip(y_predicted_values,y_test):
    comparison.append([predicted_value,true_value])
print(comparison)
# create the confusion matrix and visualize it
plt.figure(figsize=(5,5))
conf_matrix = confusion_matrix(y_test,y_predicted_values)
sns.heatmap(conf_matrix,square=True,cbar=True,annot=True,annot_kws={'size':8},cmap='Blues')


# Save the model
pk.dump(Model,open('trained_model.sav','wb'))


# Make a predictive system 
image_path = input("Enter the image path: ")
image = cv2.imread(image_path)
# Ensure the image has 3 color channels (e.g., convert from grayscale to RGB)
if image.shape[-1] == 1:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# Resize the image
resized_image = cv2.resize(image,(Image_size,Image_size))
# Normalize the image by scaling it
resized_image = resized_image / 255

# Make the model predict what is in the image if it's dog will print 1 otherwise will print 0
prediction = Model.predict(np.expand_dims(resized_image, axis=0))
predicted_class = np.argmax(prediction)
print(predicted_class)
