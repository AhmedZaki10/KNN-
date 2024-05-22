#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
import warnings 
warnings.filterwarnings("ignore")
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# Load the dataset 
train_dir = 'D:\\Work\\FCDS\\Semester 5\\DSTools\\train'
test_dir = 'D:\\Work\\FCDS\\Semester 5\\DSTools\\test'

# Generate batches of augmented/normalized data 
datagen = ImageDataGenerator(rescale=1./255, 
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), 
                                             class_mode='binary', batch_size=32)
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), 
                                            class_mode='binary', batch_size=32)

# Get training images and labels
train_images, train_labels = next(train_generator) 

# Reshape to 2D for KNN 
train_images = train_images.reshape(-1, 224*224*3)

# Apply feature scaling
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)

# Split into train and val 
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, 
                                                                      test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=15)
train_images = pca.fit_transform(train_images)
val_images = pca.transform(scaler.transform(val_images))

# Train KNN model 
knn = KNeighborsClassifier(n_neighbors=1)  
knn.fit(train_images, train_labels)

# Make predictions on val and test set
val_preds = knn.predict(val_images)

# Evaluate model 
val_accuracy = accuracy_score(val_labels, val_preds)
print("Val accuracy: {:.2f}%".format(val_accuracy*100)) 

# Cross Validation
scores = cross_val_score(knn, train_images, train_labels, cv=6)
print("Cross-validated scores:", scores)
print("Mean CV accuracy:", np.mean(scores))

# Make predictions on the whole test set
test_preds = []
test_labels = []

for i in range(len(test_generator)):
    images, labels = next(test_generator)
    images = images.reshape(-1, 224*224*3)
    images = pca.transform(scaler.transform(images))
    preds = knn.predict(images)
    
    test_preds.extend(preds)
    test_labels.extend(labels)

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(test_labels, test_preds)
print("Test accuracy: {:.2f}%".format(test_accuracy*100)) 

# Confusion matrix 
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True,fmt='g')
plt.show()

# Generate a classification report
report = classification_report(test_labels, test_preds)

# Print the report
print(report)


# In[3]:


# Fit PCA on your training data
pca = PCA().fit(train_images)

# Plot the cumulative sum of explained variance ratio
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()


# In[4]:


from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters to search over
param_grid = {
    'n_neighbors': [1, 3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Create a grid search object
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)

# Fit the grid search object to the training data
grid_search.fit(train_images, train_labels)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best hyperparameters:", best_params)

