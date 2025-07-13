# 🚁 Data Science Seminar – Semantic Segmentation with DNNs

## 📚 Project Overview

This project presents a comparative analysis of deep neural networks (VGG16 and VGG19) for semantic segmentation of aerial imagery. Conducted as part of a Data Science Seminar course, the research explores model performance in terms of accuracy and loss, and provides a complete pipeline from data preprocessing to model training, evaluation, and prediction.

Environment: Google Colab (A100 GPU, 40GB Memory) with TensorBoard monitoring.

# 🧐 Abstract

This study provides an in-depth exploration of digital image processing and computer vision techniques, beginning with the physics of image acquisition and advancing through foundational algorithms and modern deep learning approaches. It investigates the use of VGG16 and VGG19 architectures in solving semantic segmentation tasks, using aerial images captured over Dubai.

Through comprehensive experimentation with varying learning rates, VGG19 achieved superior results, reaching a test accuracy of 83.01%, outperforming VGG16's 82.51%. This improvement is largely attributed to VGG19's deeper architecture, enabling better representation and segmentation of complex spatial features in aerial imagery.

# 🔧 Methodology & Implementation

## 📦 Dataset

Source: Semantic Segmentation of Aerial Imagery (Kaggle)

Content: 72 satellite images of Dubai (MBRSC satellites), annotated for 6 semantic classes

Structure: Organized into six large tiles

## 🧰 Data Preprocessing

A custom DataGenerator module was developed to prepare the dataset:

Crops and resizes images and masks to 224×224 patches

Applies augmentations (rotation and flips, 8x per sample)

Converts masks to one-hot encoded vectors

Splits data into training (16,776), validation (5,444), and test sets (4,622)

Maintains balanced class distribution across splits

This preprocessing pipeline ensures high variability and generalization capacity across different spatial contexts.

# 🧪 Model Fine-Tuning & Training

## 🔍 Model Architecture

A custom SegmentationModel module was implemented:

Modifies VGG16 and VGG19 using torchvision backbones

Replaces classifier head with an upsampling module (5 convolutional + ReLU layers)

Outputs probability maps for each segmentation class

## 🏋️ Training & Validation

The TrainValModels module performs:

Training with various learning rates: 0.00003, 0.0001, 0.0003, 0.001, 0.003

Batch sizes: 64 (train), 32 (validation)

10 epochs per run with TensorBoard tracking (loss & accuracy)

Automatic saving of best model weights

Final evaluation on test set (batch size 16)

## 🔮 Predictions

The PredictModels module:

Loads the best-performing weights

Generates predictions on test data

Supports bypassing weight files to re-initialize models if needed

This modular setup ensures reproducibility, robustness, and a clear understanding of each model's behavior under identical training conditions.

# 📊 Key Findings

Best model: VGG19 (Test Accuracy: 83.01%)

Additional convolutional layers in VGG19 improved segmentation granularity

Custom upsampling head yielded strong pixel-wise classification performance

# 📧 Contact Me

I'm always open to feedback, collaboration, and new research opportunities.

📬 Email: maornanibar@gmail.com💬 LinkedIn: linkedin.com/in/maornave🧠 Research Interests:Computer Vision – Advanced image processing and physical simulationsData Analytics – Statistical analysis and database optimizationAI Applications – Machine & Deep learning
