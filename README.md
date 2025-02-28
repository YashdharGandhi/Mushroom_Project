**Mushroom Edibility Prediction Project**


Overview
This project presents a comprehensive analysis of mushroom edibility using machine learning techniques. The study employs decision trees and random forests to classify mushrooms as edible or poisonous based on their physical characteristics. Through rigorous model tuning, cross-validation, and statistical testing, the project identifies the most effective predictive approach.

Data Description
The analysis is based on a well-known mushroom dataset that includes several key attributes:

Edible: Indicates whether a mushroom is edible or poisonous.
CapShape: The shape of the mushroom cap.
CapSurface: The surface texture of the cap.
CapColor: The color of the cap.
Odor: The scent of the mushroom.
Height: The height of the mushroom.
Tasks Performed
Exploratory Data Analysis: An in-depth examination of the dataset was conducted to understand the distribution and relationships among features.
Model Development:
Decision Trees: A decision tree model was developed and fine-tuned using cross-validation to manage complexity and prevent overfitting.
Random Forests: A random forest model was built by optimizing parameters such as the number of trees and features considered at each split.
Model Evaluation:
Both models were assessed using key performance metrics, including accuracy, precision, recall, and F1 score.
A paired t-test was performed to statistically compare the performance of the decision tree and random forest models, ensuring that the observed differences were significant.
Key Findings
Superior Performance of Random Forests: The random forest model consistently outperformed the decision tree model across all performance metrics.
High Classification Accuracy: The random forest achieved near-perfect accuracy, precision, recall, and F1 score.
Statistical Significance: The paired t-test confirmed that the improvements in performance metrics were statistically significant, validating the efficacy of the random forest approach for predicting mushroom edibility.
Requirements
Software:
R (version 3.6 or higher recommended)
RStudio for code development and execution
