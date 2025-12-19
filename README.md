Mapping the Cosmos: Classifying Astronomical Objects from SDSS DR17

An end-to-end Machine Learning–powered astronomical classification system that automatically 
classifies celestial objects into Stars, Galaxies, and Quasars
using photometric and spectroscopic data from the Sloan Digital Sky Survey (SDSS DR17).

This project was developed as a Semester 3 Pinnacle Project for the B.Tech AI & ML program.

Project Overview

Modern sky surveys generate millions of celestial observations, 
making manual classification impractical and error-prone. 
Traditional rule-based approaches fail due to overlapping photometric properties across object classes.
This project solves the problem by:
Training and comparing 7 supervised ML models

Dataset

Source: Sloan Digital Sky Survey – Data Release 17 (via Kaggle)
Size: 100,000 astronomical objects
Target Classes:
STAR
GALAXY
QSO (Quasar)
Core Features Used
Photometric magnitudes: u, g, r, i, z
Spectroscopic redshift
Positional coordinates (RA, Dec)
Administrative identifiers were removed during preprocessing.

Machine Learning Models Used

The following classifiers were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Gaussian Naive Bayes

Support Vector Machine (RBF Kernel)

Random Forest

XGBoost


Model Evaluation:
80-20 Train-Test Split
5-Fold Cross Validation to address overfitting
Metrics: Accuracy, Precision, Recall, F1-Score
Best Performing Models:
Random Forest
XGBoost
(Achieved accuracies up to ~98%)
