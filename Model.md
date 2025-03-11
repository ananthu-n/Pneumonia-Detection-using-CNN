## Model Architecture (CNN for Pneumonia Detection)
The project implements a Convolutional Neural Network (CNN) for classifying chest X-ray images into Pneumonia and Normal categories.

### Model Summary:
- **Input Layer:** Accepts chest X-ray images (resized to a fixed dimension).  
- **Convolutional Layers:** Extract important features from the images using filters.  
- **Batch Normalization & Dropout:** Prevent overfitting and stabilize training.  
- **Max-Pooling Layers:** Reduce the spatial dimensions and retain important information.  
- **Fully Connected (Dense) Layers:** Process extracted features to make predictions.  
- **Output Layer:** Uses a sigmoid activation function to classify images as Pneumonia (1) or Normal (0).  

###Code for Model Definition:
```
python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (Pneumonia/Normal)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()
```
