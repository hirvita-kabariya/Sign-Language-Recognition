# Sign Language Recognition Model

## Project Overview
This project implements a real-time Indian Sign Language (ISL) recognition system using MediaPipe for pose estimation and hand tracking, coupled with an LSTM neural network for gesture classification.

## Features
- Real-time sign language detection using webcam input
- Recognition of 5 ISL signs: 'Hello', 'Good-Morning', 'Sorry', 'Friend', 'How-Are-You'
- Advanced pose estimation and hand tracking using MediaPipe Holistic
- LSTM-based deep learning model for sequential gesture prediction

## Technologies Used
- Python 3.x
- TensorFlow 2.5.1
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Matplotlib

## Model Architecture
- Three-layer LSTM neural network 
- Input shape: 30 frames x 1662 features (combined pose, face, and hand landmarks)
- LSTM layers with 64, 128, and 64 units respectively
- Dense output layers with softmax activation for classification
- Training with categorical cross-entropy loss

## Project Structure
- `ACC for ISL.ipynb`: Jupyter notebook containing the complete implementation
- `action.h5`: Trained LSTM model weights
- `0.npy`: Sample keypoints data file

## Installation
1. Clone this repository
2. Install the required dependencies:
```
pip install tensorflow==2.5.1 tensorflow-gpu==2.5.1 opencv-python mediapipe sklearn matplotlib
```

## Usage
1. Run the Jupyter notebook `ACC for ISL.ipynb`
2. For data collection, execute the collection cells to capture your own sign language dataset
3. For training, run the model training cells
4. For real-time prediction, execute the testing cells

## Performance
The model achieves high accuracy in recognizing the implemented sign language gestures, with confusion matrices and accuracy metrics available in the notebook.

## Future Work
- Expand the vocabulary to include more ISL signs
- Optimize the model for better performance on lower-end hardware
- Create a standalone application for easier usage
- Implement translation capabilities from sign language to text/speech
