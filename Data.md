## Dataset Information
The dataset consists of 5,863 chest X-ray images divided into train, test, and validation sets.

**Classes:**

- Pneumonia: X-ray images showing pneumonia-infected lungs.
- Normal: X-ray images of healthy lungs.
- Dataset Source:

The images are from Guangzhou Women and Children’s Medical Center.

The dataset has undergone quality control and validation by expert radiologists.

## Dataset Structure:
```
chest_xray/
│── train/
│   ├── PNEUMONIA/
│   ├── NORMAL/
│── test/
│   ├── PNEUMONIA/
│   ├── NORMAL/
│── val/
│   ├── PNEUMONIA/
│   ├── NORMAL/
```

**Loading Dataset in Google Colab:**
```
python

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

# Apply ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images into generators
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode='binary')
```
