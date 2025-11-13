# EXP 04: Transfer Learning concept in Image Classification
## Name : \KAVIPRIYA SP
## Reg.no: 2305002011
## AIM:
To implement Transfer Learning using a pre-trained MobileNetV2 model for image classification on the CIFAR-10 dataset, and evaluate the model’s performance in terms of accuracy.

## DESIGN STEPS:
Import necessary libraries and load the CIFAR-10 dataset for training and testing.

Preprocess the dataset by normalizing image pixel values to the range [0,1].

Load the pre-trained MobileNetV2 model with ImageNet weights and freeze its base layers.

Add custom classification layers suitable for CIFAR-10 image classification.

Compile and train the model using the training dataset and validate with the test dataset.

Evaluate and save the trained model to measure accuracy and store it for future use.

## PROGRAM:
``` python
# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 2: Load and Preprocess Dataset (CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to range [0,1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Load Pre-trained Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base layers

# Step 4: Add Custom Classification Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes in CIFAR-10
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))
<img width="1094" height="216" alt="Screenshot 2025-11-13 133956" src="https://github.com/user-attachments/assets/0ff2d2b8-6b94-423c-aa30-48e098deef80" />

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the Model
model.save("transfer_learning_model.h5")
print("\nModel saved successfully!")

```

## OUTPUT:
<img width="693" height="78" alt="Screenshot 2025-11-13 134011" src="https://github.com/user-attachments/assets/995ce315-f14e-4601-ae36-a7a51fda49f8" />
<img width="1094" height="216" alt="Screenshot 2025-11-13 133956" src="https://github.com/user-attachments/assets/4c16fd7f-0518-4b7c-a08c-a02efdbfc081" />


## RESULT:
The transfer learning model was successfully implemented using MobileNetV2.

The model was trained for 5 epochs and achieved an approximate test accuracy between 75% to 85%, depending on system and training conditions.

The trained model was saved as transfer_learning_model.h5 for future predictions.
