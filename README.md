# Project #2 – Ironhack

## Convolutional Neural Networks (CNNs) with CIFAR-10

---

## Project Overview

This project focuses on designing, training, and evaluating a **Convolutional Neural Network (CNN)** to accurately classify images from the **CIFAR-10** dataset.  
The work follows an **iterative modeling approach**, starting from a baseline CNN and progressively improving performance through architectural and training refinements.

The goal is not only high accuracy, but also a clear understanding of how design choices impact model behavior.

---

## Dataset

**CIFAR-10** is a standard benchmark dataset for image classification tasks. It contains:

- **60,000 color images** (32×32 pixels)
- **10 classes**
- 50,000 training images
- 10,000 test images

Each image belongs to exactly one class.

---

## Objectives

- Build a CNN from scratch using **TensorFlow / Keras**
- Develop a clean and reusable training pipeline
- Iteratively improve model performance
- Evaluate generalization on unseen test data
- Document modeling decisions and results

---

## Project Structure

project-root/
│
├── presentation/
│ └── (slides and visual material)
│
├── main.ipynb # main experimentation notebook
├── config.py # global configuration and constants
├── data.py # dataset loading and preprocessing
├── model.py # CNN architecture definitions
├── train.py # training loop and callbacks
├── metrics.py # evaluation metrics and plots
│
└── README.md

This modular structure separates concerns and improves readability, reuse, and experimentation speed.

---

## Project Members

- Edwin Santiago
- Marcos Sousa
- Rishi

---

## Deliverables

- Trained CNN model for CIFAR-10 classification
- Evaluation metrics (accuracy, loss, confusion matrix)
- Experimentation notebook with results and analysis
- Clean, modular training pipeline
- Group Presentation (.pptx)

---

## Methodology

1. Data loading and normalization
2. Baseline CNN implementation
3. Iterative architectural improvements
4. Hyperparameter tuning
5. Performance evaluation on test data

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## Results

_(To be updated)_  
Final model performance and evaluation metrics will be documented here.

---

## Reproducibility

- Fixed random seeds for deterministic results
- All dependencies listed in `requirements.txt`
- Modular pipeline for repeatable experiments

---

## Future Improvements

- Data augmentation
- Transfer learning (e.g. MobileNet, EfficientNet)
- Regularization techniques (Dropout, BatchNorm)
- Automated hyperparameter tuning

---

## Key Takeaways

- CNN performance is highly sensitive to architectural choices
- Iterative experimentation leads to more robust models
- Clean project structure improves research velocity

Note: Jupyter notebook outputs are stripped automatically during commits for repository hygiene. This does not affect execution or evaluation.

