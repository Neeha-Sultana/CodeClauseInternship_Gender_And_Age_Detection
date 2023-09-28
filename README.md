# Gender_Age_Detection_CodeClause

# Dataset Information

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.

The objective of the project is to detect gender and age using facial images. Convolutional Neural Network is used to classify the images. There are 2 output types namely, gender(M or F) and age.

**Download link:** https://www.kaggle.com/datasets/jangedoo/utkface-new

**Environment:** kaggle

# Libraries

- pandas
- numpy
- matplotlib
- keras
- tensorflow
- scikit-learn

# Neural Network

- CNN Network


# About the Project
This project is a machine learning model that aims to predict the gender and age of individuals from facial images. Here's a brief overview of the key components and steps:

1. **Importing Modules:**
   - Various Python libraries are imported, including pandas, numpy, os, matplotlib, seaborn, warnings, tqdm, and TensorFlow/Keras for deep learning.

2. **Loading the Dataset:**
   - The UTKFace dataset is loaded, containing images labeled with age, gender, and ethnicity. The dataset is organized in a directory structure.

3. **Exploratory Data Analysis (EDA):**
   - Initial exploration of the dataset is performed using visualization tools like Matplotlib and Seaborn.
   - Displaying a sample image and creating histograms for age distribution and a count plot for gender distribution.

4. **Feature Extraction:**
   - Image preprocessing and feature extraction are carried out using the Pillow library.
   - Images are resized to 128x128 pixels and converted to grayscale.
   - Features are normalized and shaped appropriately for model input.

5. **Model Creation:**
   - A convolutional neural network (CNN) model is built using the Keras Sequential API.
   - The model has convolutional layers followed by max-pooling layers and fully connected layers for gender and age prediction.
   - The model is compiled with binary crossentropy loss for gender and mean absolute error (MAE) loss for age.

6. **Training the Model:**
   - The model is trained on the preprocessed images with gender and age labels.
   - Training results, including accuracy and loss, are plotted for both gender and age.

7. **Prediction with Test Data:**
   - Several test images are selected, and the trained model is used to predict gender and age.
   - The predictions are compared with the actual labels, and the images are displayed with their predictions.




  
**Gender Accuracy:** 90.00
**Age MAE:** 6.5


