# Mushroom Edibility Prediction Project

## Overview

This project presents a comprehensive analysis of mushroom edibility using advanced machine learning techniques. By leveraging decision trees and random forests, the study classifies mushrooms as either edible or poisonous based on several morphological features. The analysis employs robust model tuning, cross-validation, and rigorous statistical testing to identify the optimal predictive approach.

## Data Description

The dataset used in this project contains key attributes that describe various characteristics of mushrooms, including:

- **Edible**: Indicates whether a mushroom is edible or poisonous.
- **CapShape**: Describes the shape of the mushroom cap.
- **CapSurface**: Details the surface texture of the cap.
- **CapColor**: Specifies the color of the cap.
- **Odor**: Captures the scent of the mushroom.
- **Height**: Measures the height of the mushroom.

## Tasks Performed

- **Exploratory Data Analysis (EDA):**  
  Detailed examination of the dataset to uncover patterns and relationships among features.

- **Model Development:**  
  - **Decision Trees:** Developed and fine-tuned a decision tree model, focusing on managing complexity and preventing overfitting.
  - **Random Forests:** Built a random forest model by optimizing key parameters such as the number of trees and the number of features considered at each split.

- **Model Evaluation:**  
  Both models were evaluated using performance metrics including accuracy, precision, recall, and F1 score. A paired t-test was employed to statistically compare the models, confirming that the performance differences were significant.

## Key Findings

- **Superior Model Performance:**  
  The random forest model demonstrated significantly higher accuracy, precision, recall, and F1 score compared to the decision tree model.

- **Statistical Validation:**  
  The paired t-test results confirmed that the improvements observed with the random forest approach were statistically significant, underscoring its effectiveness in predicting mushroom edibility.

## Requirements

- **Software:**
  - R (version 3.6 or higher recommended)
  - RStudio for development and execution

- **Required R Packages:**
  - `caret`
  - `rpart`
  - `rpart.plot`
  - `randomForest`

## Author 
**Yashdhar Gandhi**

## License
This project is licensed under the MIT License.
