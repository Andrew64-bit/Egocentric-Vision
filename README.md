# Multimodal Egocentric Action Recognition Using RGB and EMG Signals

## Abstract

This project explores egocentric action recognition using RGB video streams and ElectroMyoGraphy (EMG) signals. We utilize pre-trained models for feature extraction and implement a Variational AutoEncoder (VAE) to map visual inputs to EMG signals, enhancing action recognition systems and reconstructing missing EMG data from visual inputs.

## Introduction

Egocentric vision, capturing images or videos from the user’s perspective, has significant applications in assistive robotics, autonomous driving, industrial operations, and augmented reality. This project aims to enhance egocentric action recognition by integrating RGB and EMG signals.

## Applications

- **Assistive Robotics**: Enhances robot understanding from a human perspective.
- **Autonomous Driving**: Improves prediction and response capabilities by learning from human behavior.
- **Industrial Applications**: Enhances efficiency and accuracy in data collection and monitoring.
- **Augmented Reality**: Provides an immersive and interactive experience by overlaying digital information.

## Research Objectives

1. Extract features using a pre-trained model and train a classifier for first-person action recognition.
2. Design a model to reconstruct EMG data from RGB streams.
3. Augment egocentric vision datasets by generating EMG signals from visual data.

## Approach and Contributions

- Utilized the I3D model to extract RGB features.
- Employed models like MLP, LSTM, and TRN for classification.
- Implemented a VAE to reconstruct EMG data from RGB streams.
- Developed a method to generate EMG signals from RGB inputs.
- Conducted a comprehensive analysis and validation of the proposed multimodal approach.

## Datasets and Preprocessing

### Datasets

- **EPIC-KITCHENS**: Contains RGB video frames, optical flow, and audio data for kitchen activities.
- **ActionNet**: Includes RGB video frames and EMG signals, capturing muscle activity during various actions.

### Preprocessing

- **EMG Data**: Rectification, low-pass filtering, and normalization.
- **RGB Data**: Extracted using the I3D model with techniques like multi-scale cropping, random horizontal flipping, stacking, and normalization.

## Model Architectures

- **Temporal Recurrent Network (TRN)**: Captures multi-scale temporal relationships in video data.
- **Inception I3D**: Extracts spatio-temporal features from RGB frames.
- **Long Short-Term Memory (LSTM)**: Captures temporal dependencies in sequential data.
- **Fully Connected Variational Autoencoder (FC-VAE)**: Encodes input data into a lower-dimensional latent space and decodes it back.

## Experiments

### Setup

- **Dataset and Data Split**: Used EPIC-KITCHENS for RGB data and ActionNet for EMG data.
- **Metrics**: Cross-Entropy Loss, Accuracy, normalized Mean Squared Error (MSE), and minimal KL divergence term.

### Sampling Methods

Analyzed features using k-means clustering and visualization techniques like t-SNE and UMAP. Explored uniform and dense sampling techniques with various numbers of frames.

### VAE Training and Evaluation

- **Pre-training**: On EPIC-KITCHENS RGB features and ActionNet EMG data.
- **Fine-tuning**: Using paired RGB and EMG data from ActionNet.

### Classification Experiments

Conducted using models like MLP, LSTM, and TRN to evaluate the quality of generated EMG signals against original RGB signals.

## Results

- **MLP with Dropout**: Highest accuracy achieved with a combination of RGB and EMG embeddings.
- **LSTM**: Improved performance with a combination of RGB and EMG embeddings.
- **TRN**: Best performance observed with a balanced integration of RGB and EMG signals.

## Conclusion

This study demonstrates that integrating RGB and EMG signals enhances the performance of egocentric action recognition models. The multimodal approach captures both visual and physiological aspects, improving robustness and accuracy. Future work will focus on refining the VAE model and exploring additional modalities.

## References

Refer to the paper for detailed references.