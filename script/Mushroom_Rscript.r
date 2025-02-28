## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----Data Description and preprocesing, echo=FALSE------------------------------------------
# Load necessary libraries
library(tidyverse)
library(ggplot2)

# Load the dataset
mushrooms <- read.csv("mushrooms.csv")

# Display the structure of the dataset
str(mushrooms)

# Check for missing values
missing_values <- colSums(is.na(mushrooms))
print(missing_values)


summary(mushrooms)

# Calculate counts of edible and poisonous mushrooms
edible_count <- sum(mushrooms$Edible == "Edible")
poisonous_count <- sum(mushrooms$Edible == "Poisonous")
total_count <- nrow(mushrooms)
custom_colors <- c("cornflowerblue", "grey") 
# Create a pie chart with customizations
pie_plot <-ggplot(mushrooms, aes(x = "", fill = Edible)) +
  geom_bar(width = 1, color = "black", size = 1) +
  coord_polar(theta = "y") +
  labs(title = "Edible vs Poisonous Mushrooms",
       fill = "Edibility") +
  scale_fill_manual(values = custom_colors) + 
  theme_void() +
  theme(legend.position = "bottom")

pie_plot +
  annotate("text", x = 0, y = 0, label = paste("Edible:", edible_count), 
           size = 4, color = "black", hjust = 1.6) +
  annotate("text", x = 0, y = -0.1, label = paste("Poisonous:", poisonous_count), 
           size = 4, color = "black", hjust = -0.65) +
  annotate("text", x = 0, y = -0.2, label = paste("Total:", total_count), 
           size = 4, color = "black", hjust = 0.5)




## ----Decision Tree, echo=FALSE--------------------------------------------------------------
# Load necessary libraries
library(caret)
library(rpart)
library(rpart.plot)

# Tune decision tree model for maximal predictive performance
set.seed(123)
dt_model <- train(Edible ~ ., 
                  data = mushrooms, 
                  method = "rpart", 
                  trControl = trainControl(method = "cv", number = 20),
                  tuneLength = 20)  # Tune parameters for maximal predictive performance

# Train decision tree model using the best tuned parameters
final_dt_model <- rpart(Edible ~ ., 
                        data = mushrooms, 
                        method = "class", 
                        control = rpart.control(cp = dt_model$bestTune[["cp"]]))  # Use the best tuning parameter

# Visualize the final decision tree
# Visualize the decision tree with explanation
rpart.plot(final_dt_model, type = 2, fallen.leaves = TRUE, extra = 101)


# Assess model performance
dt_predictions <- predict(final_dt_model, mushrooms, type = "class")
dt_correctly_classified <- sum(dt_predictions == mushrooms$Edible)

# Print accuracy
print(paste("Decision Tree Accuracy:", dt_correctly_classified))



## ----random forest, echo=FALSE--------------------------------------------------------------
# Load necessary library
library(randomForest)

# Convert Edible to factor
mushrooms$Edible <- as.factor(mushrooms$Edible)

# Tune the model using cross-validation
set.seed(123)
rf_model <- train(Edible ~ ., 
                  data = mushrooms, 
                  method = "rf", 
                  trControl = trainControl(method = "cv", number = 10),
                  tuneLength = 10)  # Tune parameters for maximal predictive performance

final_rf_model <- randomForest(Edible ~ ., 
                               data = mushrooms, 
                               ntree = 500,  # Use the best tuned number of trees
                               mtry =5)  # Use the best tuned number of features to consider
# Predictions
rf_predictions <- predict(final_rf_model, mushrooms)

# Calculate accuracy
rf_accuracy <- mean(rf_predictions == mushrooms$Edible)
rf_correct_predictions <- sum(rf_predictions == mushrooms$Edible)

# Print accuracy and number of correct predictions
print(paste("Random Forest Accuracy:", rf_accuracy))
print(paste("Number of Correct Predictions:", rf_correct_predictions))

# Calculate feature importance
feature_importance <- importance(final_rf_model)

# Plot feature importance
varImpPlot(final_rf_model)




## ----Cross Validation, echo= FALSE----------------------------------------------------------
# Load necessary libraries
library(caret)
library(rpart)
library(randomForest)

# Define the number of folds for cross-validation
num_folds <- 10  # Example: 10-fold cross-validation

# Define a function to perform cross-validation for a given model
perform_cross_validation <- function(model, data, num_folds) {
  # Create a list to store performance metrics for each fold
  performance_metrics <- list()
  
  # Perform k-fold cross-validation
  folds <- createFolds(data$Edible, k = num_folds, list = TRUE)
  for (i in 1:num_folds) {
    # Split data into training and validation sets
    train_indices <- unlist(folds[-i])
    validation_indices <- unlist(folds[i])
    train_data <- data[train_indices, ]
    validation_data <- data[validation_indices, ]
    
    # Train the model on the training data
    model_fit <- train(Edible ~ ., data = train_data, method = model)
    
    # Make predictions on the validation data
    predictions <- predict(model_fit, newdata = validation_data)
    
    # Calculate performance metrics
    accuracy <- sum(predictions == validation_data$Edible) / length(predictions)
    confusion_matrix <- confusionMatrix(predictions, validation_data$Edible)
    precision <- confusion_matrix$byClass["Precision"]
    recall <- confusion_matrix$byClass["Recall"]
    f1_score <- confusion_matrix$byClass["F1"]
    
    # Store performance metrics for this fold
    performance_metrics[[i]] <- c(accuracy, precision, recall, f1_score)
  }
  
  return(do.call(rbind, performance_metrics))
}

# Perform cross-validation for decision tree model
dt_performance <- perform_cross_validation("rpart", mushrooms, num_folds)

# Perform cross-validation for random forest model
rf_performance <- perform_cross_validation("rf", mushrooms, num_folds)

# Print all the values necessary for calculating Performance Metrics
print("Decision Tree Performance Metrics:")
print(dt_performance)
print("Random Forest Performance Metrics:")
print(rf_performance)

# Calculate mean performance metrics for decision tree
dt_mean_performance <- colMeans(dt_performance)
print("Mean Performance Metrics for Decision Tree:")
print(dt_mean_performance)

# Calculate mean performance metrics for random forest
rf_mean_performance <- colMeans(rf_performance)
print("Mean Performance Metrics for Random Forest:")
print(rf_mean_performance)

# Perform paired t-test on Accuracy
t_test_accuracy <- t.test(dt_performance[, 1], rf_performance[, 1], paired = TRUE)
print("Paired t-test Results for Accuracy:")
print(t_test_accuracy)

# Perform paired t-test on Precision
t_test_precision <- t.test(dt_performance[, 2], rf_performance[, 2], paired = TRUE)
print("Paired t-test Results for Precision:")
print(t_test_precision)

# Perform paired t-test on Recall
t_test_recall <- t.test(dt_performance[, 3], rf_performance[, 3], paired = TRUE)
print("Paired t-test Results for Recall:")
print(t_test_recall)

# Perform paired t-test on F1 Score
t_test_f1 <- t.test(dt_performance[, 4], rf_performance[, 4], paired = TRUE)
print("Paired t-test Results for F1 Score:")
print(t_test_f1)


