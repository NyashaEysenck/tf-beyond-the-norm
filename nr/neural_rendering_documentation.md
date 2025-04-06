# Neural Rendering for Low-Bandwidth Video Calls

## Overview
This notebook demonstrates a neural rendering system for low-bandwidth video calls. It uses facial keypoint detection on cropped face images with 68 landmark points to train a deep learning model that can encode facial movements as lightweight neural weights, dramatically reducing bandwidth requirements from 5MB/sec to just 5KB/sec.

## Application
- **Low-Bandwidth Video Calls**: The model encodes facial movements as lightweight neural weights
- **Real-time Reconstruction**: The receiver's device reconstructs the face in real-time using a tiny generative model
- **Bandwidth Efficiency**: Achieves 1000x bandwidth reduction (5KB/sec vs 5MB/sec for traditional video)

## Technology
- TensorFlow Graphics
- Differentiable Rendering
- Neural encoding/decoding

## Dataset
The notebook uses the "Cropped Face Keypoint Dataset (68 Landmarks)" from Kaggle, which contains:
- Training and test images of cropped faces
- CSV files with 68 facial keypoints (x,y coordinates) for each image
- Each keypoint represents a specific facial feature (e.g., corners of eyes, nose tip, mouth corners)

## Notebook Structure

### 1. Setup and Data Loading
- Installation of required packages
- Downloading the dataset from Kaggle using `kagglehub`
- Exploring the dataset structure and contents
- Loading and inspecting the training CSV file which contains image names and keypoint coordinates

### 2. Data Visualization
- Displaying sample images with their corresponding facial keypoints
- Visualizing how the keypoints map to facial features

### 3. Data Preprocessing
- Normalizing the landmarks (centering and scaling to [0, 1])
- Loading and preprocessing images to a standard size (128x128)
- Splitting data into training and validation sets
- Preparing data for efficient encoding of facial movements

### 4. Model Architecture
The model follows an encoder-decoder architecture optimized for bandwidth efficiency:

#### Encoder (Sender Side)
- Based on MobileNetV2 (pre-trained on ImageNet)
- Takes facial image frames as input and encodes them into compact latent vectors
- Extracts only essential facial movement data (keypoints)
- Compresses information to achieve the 5KB/sec target bandwidth

#### Decoder (Receiver Side)
- Takes the lightweight neural weights and reconstructs the facial image
- Consists of several dense layers with increasing dimensions
- Outputs 136 values (68 keypoints × 2 coordinates)
- Feeds into a generative model that renders the complete face

### 5. Model Training
- Compiling the model with appropriate loss function (MSE) and optimizer
- Training the model to minimize both reconstruction error and bandwidth usage
- Monitoring training progress and validation performance
- Fine-tuning for real-time performance on mobile devices

### 6. Evaluation and Visualization
- Evaluating the model on validation data
- Measuring bandwidth efficiency (target: 5KB/sec)
- Visualizing reconstructed faces against original video frames
- Analyzing latency and quality tradeoffs

### 7. Real-time Implementation
- Converting the model for mobile deployment
- Implementing the sender-side encoding pipeline
- Creating the receiver-side reconstruction system
- Testing end-to-end performance in simulated video call scenarios

## Usage Instructions
1. Run the cells in order from top to bottom
2. Make sure you have a GPU runtime enabled for faster training
3. The dataset will be downloaded automatically using kagglehub
4. The model training may take some time depending on your hardware
5. Test the bandwidth efficiency by measuring the size of encoded facial movements
6. Experiment with different compression rates to find the optimal quality/bandwidth balance

## Requirements
- TensorFlow 2.x and TensorFlow Graphics
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- kagglehub (for dataset download)

## Potential Applications
- Video conferencing in low-bandwidth environments
- Remote areas with limited internet connectivity
- Mobile video calls with data usage constraints
- VR/AR avatars driven by facial expressions
- Telepresence systems with minimal latency requirements

## References
- The dataset used is from Kaggle: "Cropped Face Keypoint Dataset (68 Landmarks)" by Sovit Rathod
- TensorFlow Graphics for differentiable rendering
- Research on neural compression for facial animation
