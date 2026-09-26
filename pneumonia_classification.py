# ============================================================
# PNEUMONIA vs NORMAL - CNN BINARY CLASSIFICATION
# Dataset: Chest X-Ray Images (Normal and Pneumonia)
# ============================================================

import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------------------------------------
# 1. DOWNLOAD DATASET
# ------------------------------------------------------------

path = kagglehub.dataset_download(
    "ghost5612/chest-x-ray-images-normal-and-pneumonia"
)

print("Dataset path:", path)


# ------------------------------------------------------------
# 2. FIND TRAIN / VALIDATION / TEST DIRECTORIES
# ------------------------------------------------------------

def find_directory(root, directory_name):
    for current_root, dirs, files in os.walk(root):
        if directory_name in dirs:
            return os.path.join(current_root, directory_name)
    return None


train_dir = find_directory(path, "train")
val_dir = find_directory(path, "val")
test_dir = find_directory(path, "test")

print("\nTrain:", train_dir)
print("Validation:", val_dir)
print("Test:", test_dir)


# ------------------------------------------------------------
# 3. PARAMETERS
# ------------------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


# ------------------------------------------------------------
# 4. LOAD DATA
# ------------------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("\nClasses:", class_names)


# ------------------------------------------------------------
# 5. IMPROVE DATA PIPELINE PERFORMANCE
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# 6. DATA AUGMENTATION
# ------------------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])


# ------------------------------------------------------------
# 7. CNN MODEL
# ------------------------------------------------------------

model = models.Sequential([

    layers.Input(shape=(224, 224, 1)),

    # Normalization
    layers.Rescaling(1.0 / 255),

    # Data Augmentation
    data_augmentation,

    # -------------------------
    # CNN BLOCK 1
    # -------------------------
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -------------------------
    # CNN BLOCK 2
    # -------------------------
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -------------------------
    # CNN BLOCK 3
    # -------------------------
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -------------------------
    # CLASSIFICATION HEAD
    # -------------------------
    layers.Flatten(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dense(
        16,
        activation="relu"
    ),

    # Binary Classification
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ------------------------------------------------------------
# 8. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 9. MODEL SUMMARY
# ------------------------------------------------------------

model.summary()


# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save("pneumonia_cnn_model.keras")
print("\n[INFO] Model saved as: pneumonia_cnn_model.keras")


# ------------------------------------------------------------
# 11. TEST EVALUATION
# ------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)


# ------------------------------------------------------------
# 12. PREDICTIONS
# ------------------------------------------------------------

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (predictions >= 0.5).astype(int)

    y_true.extend(labels.numpy().flatten())
    y_pred.extend(predictions.flatten())


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ------------------------------------------------------------
# 13. METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

print("\n==============================")
print("CLASSIFICATION METRICS")
print("==============================")

print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)


# ------------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ------------------------------------------------------------
# 15. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show(block=False)


# ------------------------------------------------------------
# 16. TRAINING & VALIDATION GRAPHS
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_graphs.png")
plt.show(block=False)


# ------------------------------------------------------------
# 17. SAVE MODEL
# ------------------------------------------------------------

model.save("pneumonia_cnn_model.keras")

print("\nModel saved as: pneumonia_cnn_model.keras")
