# Comparative-Analysis-of-KNN-VGGNet-and-AlexNet-for-Robust-Latent-Palm-Print-Recognition
This has a python code in which a comparison between KNN, VGGNet and AlexNet is done on a dataset of latent palm prints for biometric recognition.
#Latest Epoch 20 and K=3
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ✅ MOUNT GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')

# ✅ LOAD DATASET
dataset_path = "/content/drive/MyDrive/Separate"

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=16
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=16
)

# ✅ Extract class names
class_names = train_data.class_names

# ✅ Convert dataset into numpy arrays
def dataset_to_numpy(dataset):
    images, labels = [], []
    for image_batch, label_batch in dataset:
        images.extend(image_batch.numpy())
        labels.extend(label_batch.numpy())
    return np.array(images), np.array(labels)

X_train, y_train = dataset_to_numpy(train_data)
X_test, y_test = dataset_to_numpy(test_data)

# ✅ Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# ✅ Reshape for KNN
X_train_knn = X_train.reshape(X_train.shape[0], -1)
X_test_knn = X_test.reshape(X_test.shape[0], -1)

# ✅ KNN MODEL
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_knn, y_train)
knn_preds = knn.predict(X_test_knn)

knn_accuracy = accuracy_score(y_test, knn_preds)
knn_precision = precision_score(y_test, knn_preds, average='weighted')
knn_recall = recall_score(y_test, knn_preds, average='weighted')
knn_f1 = f1_score(y_test, knn_preds, average='weighted')

print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
print(f"KNN Precision: {knn_precision:.2f}")
print(f"KNN Recall: {knn_recall:.2f}")
print(f"KNN F1 Score: {knn_f1:.2f}")

# ✅ MANUAL EPOCH CONTROL
num_epochs = 20  # Change this value to control epochs manually

# ✅ VGGNET MODEL (Fine-Tuned)
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_vgg.trainable = True  # Fine-tune VGG16

vgg_model = Sequential([
    base_vgg,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

vgg_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_vgg = vgg_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), batch_size=16)

vgg_preds = np.argmax(vgg_model.predict(X_test), axis=-1)
vgg_accuracy = accuracy_score(y_test, vgg_preds)
vgg_precision = precision_score(y_test, vgg_preds, average='weighted')
vgg_recall = recall_score(y_test, vgg_preds, average='weighted')
vgg_f1 = f1_score(y_test, vgg_preds, average='weighted')

print(f"VGGNet Accuracy: {vgg_accuracy * 100:.2f}%")
print(f"VGGNet Precision: {vgg_precision:.2f}")
print(f"VGGNet Recall: {vgg_recall:.2f}")
print(f"VGGNet F1 Score: {vgg_f1:.2f}")

# ✅ ALEXNET MODEL
alexnet_model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(3, 3),
    Conv2D(256, (5, 5), activation='relu', padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(384, (3, 3), activation='relu', padding="same"),
    Conv2D(384, (3, 3), activation='relu', padding="same"),
    Conv2D(256, (3, 3), activation='relu', padding="same"),
    MaxPooling2D(3, 3),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

alexnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_alex = alexnet_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), batch_size=16)

alexnet_preds = np.argmax(alexnet_model.predict(X_test), axis=-1)
alexnet_accuracy = accuracy_score(y_test, alexnet_preds)
alexnet_precision = precision_score(y_test, alexnet_preds, average='weighted')
alexnet_recall = recall_score(y_test, alexnet_preds, average='weighted')
alexnet_f1 = f1_score(y_test, alexnet_preds, average='weighted')

print(f"AlexNet Accuracy: {alexnet_accuracy * 100:.2f}%")
print(f"AlexNet Precision: {alexnet_precision:.2f}")
print(f"AlexNet Recall: {alexnet_recall:.2f}")
print(f"AlexNet F1 Score: {alexnet_f1:.2f}")

# ✅ PLOT RESULTS
model_names = ["KNN", "VGGNet", "AlexNet"]
accuracies = [knn_accuracy * 100, vgg_accuracy * 100, alexnet_accuracy * 100]
precisions = [knn_precision, vgg_precision, alexnet_precision]
recalls = [knn_recall, vgg_recall, alexnet_recall]
f1_scores = [knn_f1, vgg_f1, alexnet_f1]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Accuracy Plot
axes[0].bar(model_names, accuracies, color=['blue', 'red', 'green'])
axes[0].set_title("Accuracy Comparison")
axes[0].set_ylim([50, 100])
axes[0].set_ylabel("Accuracy (%)")

# Precision Plot
axes[1].bar(model_names, precisions, color=['blue', 'red', 'green'])
axes[1].set_title("Precision Comparison")
axes[1].set_ylim([0, 1])
axes[1].set_ylabel("Precision")

# Recall Plot
axes[2].bar(model_names, recalls, color=['blue', 'red', 'green'])
axes[2].set_title("Recall Comparison")
axes[2].set_ylim([0, 1])
axes[2].set_ylabel("Recall")

# F1 Score Plot
axes[3].bar(model_names, f1_scores, color=['blue', 'red', 'green'])
axes[3].set_title("F1 Score Comparison")
axes[3].set_ylim([0, 1])
axes[3].set_ylabel("F1 Score")

plt.tight_layout()
plt.show()
