Project Description: F.R.I.D.A.Y (Facial Recognition Image Detection and Analysis System)

In this project, I developed F.R.I.D.A.Y, an application that focuses on facial recognition, image detection, and analysis.
The goal was to build a robust system capable of detecting and analyzing faces in real-time using computer vision techniques. 
Here, I will explain the step-by-step process of building the face detection model and the overall pipeline for object classification.
To begin, I utilized a webcam to collect images for training the face detection model.
However, having only around 100 images is insufficient for training a deep learning model effectively. 
To address this, I used the augmentation library to augment the images. This involved randomizing the input images by adjusting brightness, gamma value, and cropping the images around 30 times, resulting in approximately 3000 augmented images. This augmented dataset was deemed sufficient to train the model. Additionally, I used bounding boxes to annotate the faces in the images.
The face detection model consists of two main components: a classification model and a regression model. 
The classification model determines whether an image contains a face or not, while the regression model helps draw a bounding box around the face by estimating the coordinates. For the regression model, two diagonal points are required.
After training the models, the next step is to determine the loss function to adjust the hyperparameters.
Since the model comprises a classification model and a regression model, 
I used the Binary Entropy Loss for classification and Mean Squared Error (MSE) loss or localization loss for regression.
For building the neural network model for face detection, I used the Keras API. 
Specifically, I employed the VGG16 model, which is pre-trained on thousands of images.
I added the final two layers: one for classification with a sigmoid activation function, and the other for regression.
At the end of the process, five different values are obtained: one ranging between 0 and 1 for the classification model (using the sigmoid activation function) and four values for the two coordinates obtained from the regression model.

The pipeline for F.R.I.D.A.Y involves the following steps:

Setup and Data Collection:

Install dependencies and set up the environment by installing necessary libraries such as Label Me, TensorFlow, TensorFlow-GPU, OpenCV Python, Matplotlib, and Albumentations.
Collect images using OpenCV, utilizing the OS library for navigating through file paths, the Time library for timing purposes, and the UUID library for creating unique identifiers for the images.

Annotation of Images:

Annotate the collected images using the Label Me tool.
Choose the image directory and save the output in the labels directory.
Select the option to save the annotations automatically.

Data Partitioning and Augmentation:

Preprocess the data by visualizing it using Matplotlib.
Split the data into training, testing, and validation sets.
Augment the small dataset of 239 images by adjusting brightness, cropping, or other transformations to effectively train the model.

Combining Labels and Images:

Combine the labels and annotated images into a new combined dataset.

Loading and Partitioning Augmented Data:

Load the labels and images into separate datasets for training, testing, and validation.

Model Building and Pipeline Definition:

Build the face detection model, defining the pipeline along with the optimizer and loss function.

Training and Evaluation:

Pass the training and validation sets through the model and evaluate its performance per epoch.

Saving and Loading the Model:

Save the trained model for future use and load it for prediction.

Real-Time Face Detection:

Utilize the model to perform real-time face detection.

In conclusion, the F.R.I.D.A.Y project involved developing a facial recognition, image detection, and analysis system. 
The pipeline for object classification included steps such as data collection, annotation, data partitioning and augmentation, model building, training, evaluation, and real-time face detection. 
By following this pipeline, F.R.I.D.A.Y successfully detects and analyzes faces in real-time, providing a robust and efficient system for facial recognition and object classification.
