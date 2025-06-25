# Live Classifier

## 📸 Overview
This is a real-time object classification system using a Convolutional Neural Network (CNN). The system will classify objects in live video feed, frame by frame, into predefined classes such as remote control, cell phone, TV, and computer keyboard. The project involves collecting and preprocessing a dataset, designing and training a CNN model, applying the model to live webcam feed, and displaying the classified objects in real-time.

<img width="400" alt="Screenshot 2025-06-10 at 12 28 39 AM" src="https://github.khoury.northeastern.edu/jinggghui/CS5330_Su25_Group4/assets/19718/19c85ccd-2fab-46f2-8d6e-14879850b4d6">

Model trained under: https://github.khoury.northeastern.edu/jinggghui/CS5330_Su25_Group4/blob/main/mini-project8/cnn_model.keras

## 📁 Dataset 
- 📱 Cell Phone - 200 images
- 🎮 Remote Control - 200 images
- 🖥️ TV - 200 images
- ⌨️ Computer Keyboard - 200 images

All images were located in inside the datasets/raw/ directory. Each image was:

- Loaded using OpenCV
- Resized to a fixed resolution defined in Constants.py (IMG_SIZE)
- Converted from BGR to RGB color space
- Normalized to the range [0, 1]

## 🎥 Live Classification Usage
🔴 Run the Live Classifier
- python WebCamSave.py
- Launches webcam using OpenCV
- Classifies each frame
- Displays class probabilities + FPS on the screen
- Press q to quit
- Uses cnn_model.keras and Constants.py to configure input size and class names

## 🧠 CNN Model Architecture
The model was built using TensorFlow Keras, with the following architecture:

Input: (RESOLUTION, RESOLUTION, 3)

[Conv2D] 32 filters, 3x3 kernel, ReLU + L2 regularization
[MaxPooling2D] 2x2 pool

[Conv2D] 64 filters, 3x3 kernel, ReLU + L2 regularization
[MaxPooling2D] 2x2 pool

[Conv2D] 128 filters, 3x3 kernel, ReLU + L2 regularization
[MaxPooling2D] 2x2 pool

[Flatten]

[Dense] 128 units, ReLU + L2 regularization
[Dropout] 0.5

[Dense] Softmax layer (output = 4 classes) + L2 regularization

### Regularization Techniques Used:

- L2 Regularization (l2_factor = 0.001): To reduce overfitting.
- Dropout (0.5): Randomly disables 50% of neurons in the dense layer during training.

### Data Augmentation (via ImageDataGenerator):

- Rotation range: 20 degrees
- Width/Height shift: 10%
- Zoom range: 10%
- Horizontal flip: enabled
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Early Stopping: Enabled with patience=4, using validation loss as a signal.

## 📊 Model Evaluation

### Training
We trained a CNN using a balanced image dataset with 800 photos in total—200 images for each of the four classes. We used 75% of the data to train the model and 25% to validate it. 
Test accuracy was 79.5%
Best training accuracy reached 86.6%
Best validation accuracy was 82.0%
The model performed well across all four categories, with F1-scores (a measure of accuracy that includes both precision and recall) between 0.73 and 0.86. 
cell_phone class could be improved by adding more training images.

Overall, the model learned the task well and gives good results on new, unseen data.

## 🎬 Demonstration Video
Link: https://northeastern.zoom.us/rec/share/vkkdtJH0VkPtRSFkfNgVeiM68ZR87aU6dVB0xG8dWfVXHAzFiJVIWAeg8kO7l8ci.R8eanCZ4WhtYiczM
Passcode: 51N95$ji

## 🔧 Prerequisites

Ensure you have Python 3.8+ installed, along with the following libraries:

```bash
pip install opencv-python tensorflow numpy matplotlib scikit-learn
python -m venv venv
source venv/bin/activate # on Mac

