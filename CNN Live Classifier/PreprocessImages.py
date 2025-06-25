import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from Constants import *

def get_data_labels():
    class_folders = {
        CELL_PHONE: os.path.join(INPUT_FOLDER, CELL_PHONE),
        COMPUTER_KEYBOARD: os.path.join(INPUT_FOLDER, COMPUTER_KEYBOARD),
        TV: os.path.join(INPUT_FOLDER, TV),
        REMOTE_CONTROL: os.path.join(INPUT_FOLDER, REMOTE_CONTROL),
    }

    data = []
    labels = []
    filenames = []
    class_counts = defaultdict(int)

    for label, folder in class_folders.items():
        if not os.path.exists(folder):
            print(f"Warning: Folder not found for class '{label}': {folder}")
            continue

        file_list = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]

        for i, file in enumerate(file_list, start=1):
            input_path = os.path.join(folder, file)
            img = cv2.imread(input_path)
            if img is None:
                print(f'Could not read "{input_path}", skipping')
                continue

            resized_img = cv2.resize(img, IMG_SIZE)
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            normalized_img = rgb_img.astype(np.float32) / 255.0

            data.append(normalized_img)
            filenames.append(file)
            labels.append(label)
            class_counts[label] += 1

    print("Done: All images loaded, resized, and labeled.")
    print(f'Class counts {class_counts}')
    return data, labels, filenames

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def build_model(input_shape, num_classes, l2_factor=0.001):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor)),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu', kernel_regularizer=l2(l2_factor)),
        Dropout(0.5),

        Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_factor))
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def renaming(folder_name):
    FOLDER = f"datasets/raw/{folder_name}"  # adjust path as needed
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic"}

    # Get list of files with supported extensions
    files = sorted([
        f for f in os.listdir(FOLDER)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ])

    # Rename files
    for i, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"{folder_name}-{i:03d}{ext}"
        old_path = os.path.join(FOLDER, old_name)
        new_path = os.path.join(FOLDER, new_name)

        os.rename(old_path, new_path)

    print(f"Renaming complete for {folder_name}.")

def train_model(withValidation=False):
    data, labels, filenames = get_data_labels()

    # Encode the labels
    lb = LabelBinarizer()
    labels_encoded = lb.fit_transform(labels)
    print(f'Label binarizer order: {lb.classes_}')

    # Save LabelBinarizer
    with open("label_binarizer.pkl", "wb") as f:
        pickle.dump(lb, f)
    print("Saved label binarizer as 'label_binarizer.pkl'")

    # Use all data for training
    X = np.array(data)
    Y = np.array(labels_encoded)

    if withValidation:
        trainX, testX, trainY, testY, labels_train, labels_test, filenames_train, filenames_test = train_test_split(
            X, Y, np.array(labels), np.array(filenames),
            test_size=0.25, stratify=Y, random_state=42
        )
    else:
        trainX, trainY = X, Y
        testX, testY = None, None

    # Build the TensorFlow model
    model = build_model(input_shape=(RESOLUTION, RESOLUTION, 3), num_classes=len(lb.classes_))

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X)

    print("Starting training...")
    if withValidation:
        history = model.fit(datagen.flow(trainX, trainY, batch_size=32),
                            validation_data=(testX, testY),
                            epochs=30,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    else:
        history = model.fit(datagen.flow(trainX, trainY, batch_size=32),
                            epochs=30,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

    # Report best training accuracy epoch
    best_epoch = np.argmax(history.history['accuracy']) + 1
    print(f"Best training accuracy at epoch: {best_epoch}")

    # Save the model trained
    model.save("cnn_model.keras")
    print("Model trained and saved as 'cnn_model.keras'.")

    if withValidation:
        # Plot accuracy
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Over Epochs')
        plt.show()

        # Plot loss
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Over Epochs')
        plt.show()

        # Predict
        predictions = model.predict(testX)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(testY, axis=1)

        # Evaluation
        accuracy = np.mean(pred_classes == true_classes)
        print(f'Test accuracy: {accuracy * 100:.2f}%')
        print(classification_report(
            true_classes,
            pred_classes,
            labels=np.arange(len(lb.classes_)),
            target_names=lb.classes_,
            zero_division=0
        ))

        # Display predictions
        sample_indices = np.random.choice(len(testX), 20, replace=False)
        sample_images = testX[sample_indices]
        sample_predictions = pred_classes[sample_indices]
        sample_true_labels = true_classes[sample_indices]
        sample_filenames = np.array(filenames_test)[sample_indices]

        plt.figure(figsize=(30, 12))
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            plt.imshow(sample_images[i].squeeze())
            plt.title(f"{sample_filenames[i]}\nPred: {lb.classes_[sample_predictions[i]]}\nTrue: {lb.classes_[sample_true_labels[i]]}")
            plt.axis('off')
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        plt.show()


def main():
    # Run renaming for each folder
    # for folder in [CELL_PHONE, COMPUTER_KEYBOARD, TV, REMOTE_CONTROL]:
    #     renaming(folder)

    # train_model(withValidation=True)
    train_model()

if __name__ == '__main__':
    main()


