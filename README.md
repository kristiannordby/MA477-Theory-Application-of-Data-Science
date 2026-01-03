# MA477: Theory and Application of Data Science

**Course Repository - United States Military Academy, West Point**

---

## Overview

This repository contains coursework, projects, and assignments completed for **MA477: Theory and Application of Data Science** at the United States Military Academy, West Point.

MA477 is a comprehensive data science course that covers fundamental machine learning algorithms, statistical modeling, and practical applications using Python. The course emphasizes both theoretical understanding and hands-on implementation of data science techniques.

---

## Course Topics

The course covers a broad range of data science and machine learning topics, including:

**Foundational Tools:**
- Python for Data Science
- NumPy and Pandas
- Data visualization with Matplotlib and Seaborn
- Jupyter Notebooks

**Supervised Learning:**
- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Trees and Random Forests
- Naive Bayes Classification
- Model evaluation and performance metrics

**Unsupervised Learning:**
- K-Means Clustering
- Principal Component Analysis (PCA)
- Dimensionality reduction techniques

**Data Preprocessing:**
- Handling missing values
- Feature engineering and selection
- Data normalization and standardization
- Train-test splitting and cross-validation

---

## Repository Structure

```
MA477-Theory-Application-of-Data-Science/
│
├── Project1/          # First course project
├── Project2/          # Second course project
└── README.md          # This file
```

Each project folder contains Jupyter Notebooks with code, analysis, and documentation for the respective assignments.

---

## Projects

### Project 1: End-to-End Machine Learning - Used Car Price Prediction

**Objective:** Build a regression model to predict used car prices based on various vehicle attributes.

**Dataset:** 188,533 used car listings with 13 features including:
- Brand, model, and model year
- Mileage and fuel type
- Engine specifications and transmission
- Exterior and interior color
- Accident history and title status
- Price (target variable)

**Methodology:**
- Complete end-to-end ML pipeline following industry best practices
- Exploratory data analysis and visualization
- Feature engineering and data preprocessing
- Missing value imputation and categorical encoding
- Model selection and hyperparameter tuning
- Cross-validation for performance estimation

**Models Evaluated:**
- Ridge Regression (alpha=300)
- Additional regression models for comparison

**Key Techniques:**
- Train/validation/test split with stratification
- Pipeline construction for reproducibility
- Root Mean Squared Error (RMSE) optimization
- Competitive leaderboard scoring against peers

**Deliverables:**
- 3-page executive summary (LaTeX)
- Jupyter notebook with documented code
- Predictions on held-out test set

**Project Type:** Individual | **Points:** 200 | **Due:** February 2025

---

### Project 2: Binary Classification - Spaceship Titanic

**Objective:** Predict which passengers were transported to an alternate dimension during the Spaceship Titanic collision with a spacetime anomaly.

**Dataset:** 8,693 passenger records from the Spaceship Titanic manifest with features including:
- PassengerId, HomePlanet, Destination
- CryoSleep status and VIP designation
- Cabin location and Age
- Luxury amenity spending (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)
- Transported status (target variable - binary classification)

**Problem Setting:** 
Based on the Kaggle competition "Spaceship Titanic," this project involves predicting passenger outcomes using recovered data from the damaged computer system after the collision with a spacetime anomaly near Alpha Centauri.

**Methodology:**
- End-to-end classification pipeline
- Feature engineering from passenger records
- Handling missing values and categorical variables
- Model comparison and ensemble methods
- Hyperparameter optimization

**Models Evaluated:**
- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Bagging Classifier with SVM base estimator

**Performance Metrics:**
- Accuracy
- Precision, Recall, F1-Score
- Cross-validation scores
- Kaggle leaderboard ranking

**Best Model:** Bagging Classifier with SVM (Accuracy: ~78.5%)

**Deliverables:**
- 2-page government report (LaTeX) for Space Travel Safety Board
- Jupyter notebook with model development
- Kaggle submission (CSV format)

**Project Type:** Individual | **Points:** 200 | **Due:** March 2025

---

## Technical Requirements

**Programming Language:** Python 3.x

**Key Libraries:**
- NumPy - Numerical computing
- Pandas - Data manipulation and analysis
- Matplotlib/Seaborn - Data visualization
- Scikit-learn - Machine learning algorithms
- Jupyter - Interactive notebooks

---

## Usage

To view or run the notebooks in this repository:

1. Clone the repository:
```bash
git clone https://github.com/kristiannordby/MA477-Theory-Application-of-Data-Science.git
cd MA477-Theory-Application-of-Data-Science
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Navigate to the desired project folder and open the `.ipynb` files.

---

## Academic Integrity

This repository contains academic coursework completed as part of the MA477 curriculum at West Point. The code and analyses presented here represent my own work in accordance with the United States Military Academy's Honor Code.

**West Point Honor Code:**  
*"A cadet will not lie, cheat, steal, or tolerate those who do."*

If you are currently enrolled in MA477 or a similar course, please use this repository only as a reference for understanding concepts and approaches. Do not copy code or analyses directly, as this would constitute a violation of academic integrity policies.

---

## About the Course

**Institution:** United States Military Academy  
**Course:** MA477 - Theory and Application of Data Science  
**Department:** Department of Mathematical Sciences  
**Level:** Undergraduate (Upper-Division)

MA477 prepares cadets to apply data science techniques to real-world problems, developing skills in statistical reasoning, programming, and analytical thinking that are essential for modern military and civilian leadership roles.

---

## License

This repository is open-source and available for educational purposes.

All course materials, assignments, and projects are subject to West Point's academic policies. External users are welcome to review the code and methodologies for learning purposes, but should not submit this work as their own.

---

## Contact

**Author:** Kristian Nordby  
**Institution:** United States Military Academy, Class of 2026  
**GitHub:** [kristiannordby](https://github.com/kristiannordby)

For questions about this repository or the coursework contained within, please open an issue or contact me directly through GitHub.

---

## Acknowledgments

Special thanks to the MA477 instructors and the Department of Mathematical Sciences at West Point for providing a rigorous and practical introduction to data science and machine learning.

---

**Last Updated:** January 2026
