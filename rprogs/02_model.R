#!/usr/bin/env Rscript

#####################################
##    NAME: 04_xgb.R               ##
##    COMP: Kaggle Titanic Intro   ##
##  AUTHOR: Mark Roepke            ##
##    DATE: 2016-12-10             ##
## PURPOSE: Create extreme         ##
##          gradient boosting      ##
##          models                 ##
#####################################

#########
# SETUP #
#########

# PREPARE SCRIPT #
rm(list = ls())
#setwd("/mnt/c/Users/roepk_000/OneDrive/Documents/Data/data/kaggle/titanic")
setwd("C://Users//roepk_000//OneDrive//Documents//Data//data//kaggle//titanic")

# LOAD PACKAGES #
library(tidyverse)
library(xgboost)
library(caret)
library(Matrix)

#########################
# Prepare Modeling Data #
#########################

# Load Data #
combined_data <- read_rds("input/combined_data.rds")

# Create Sparse Matrices for Modeling #
train_pred_dm <- combined_data %>%
  filter(data == "train") %>%
  select(-c(data, survived)) %>%
  data.matrix()

train_class_df <-  combined_data %>%
  filter(data == "train") %>%
  select(survived) %>%
  mutate(survived = factor(survived))

test_pred_dm <- combined_data %>%
  filter(data == "test") %>%
  select(-c(data, survived)) %>%
  data.matrix()

############
# Modeling #
############

# Create hyperparameter grid
xgb_grid <- expand.grid(
  eta = seq(from = 0.2, to = 0.3, by = 0.05),
  max_depth = seq(from = 1, to = 3, by = 1),
  nrounds = seq(from = 50, to = 100, by = 5),
  min_child_weight = 1/sqrt(length(train_class_df[train_class_df$survived == 1,])/nrow(train_class_df)),
  colsample_bytree = seq(from = 0.2, to = 0.3, by = 0.05),
  gamma = 1,
  subsample = 1
)

# Set cross-validation parameters
xgb_ctrl = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE
)

# Train CV model
set.seed(1831)
xgb_train_cv <- train(
  x = train_pred_dm,
  y = train_class_df$survived,
  trControl = xgb_ctrl,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

# Check CV results
xgb_train_cv_accuracy <- tibble(
  predictions = predict(xgb_train_cv, train_pred_dm),
  actual = train_class_df$survived
) %>%
  mutate(correct = ifelse(predictions == actual, 1, 0)) %>%
  summarize(correct_rate = mean(correct))
print(str_c("CV Accuracy = ", xgb_train_cv_accuracy[["correct_rate"]][[1]]))

# Check best parameters
xgb_train_cv$bestTune

# Re-train model on all training data with bestTune parameters
set.seed(1831)
xgb_train_total <- train(
  x = train_pred_dm,
  y = train_class_df$survived,
  tuneGrid = xgb_train_cv$bestTune,
  method = "xgbTree"
)

# Check total results
xgb_train_total_accuracy <- tibble(
  predictions = predict(xgb_train_total, train_pred_dm),
  actual = train_class_df$survived
) %>%
  mutate(correct = ifelse(predictions == actual, 1, 0)) %>%
  summarize(correct_rate = mean(correct))
print(str_c("Total Accuracy = ", xgb_train_total_accuracy[["correct_rate"]][[1]]))

###############################
# Compute Holdout Predictions #
###############################

# Create cv submission tibble
titanic_submission_cv <- tibble(
  PassengerId = as_tibble(test_pred_dm)$passengerid,
  Survived = predict(xgb_train_cv, test_pred_dm)
)

# Save as CSV
fwrite(titanic_submission_cv, "output/xgb_titanic_submission_cv.csv")


# Create total submission tibble
titanic_submission_total <- tibble(
  PassengerId = as_tibble(test_pred_dm)$passengerid,
  Survived = predict(xgb_train_total, test_pred_dm)
)

# Save as CSV
fwrite(titanic_submission_total, "output/xgb_titanic_submission_tot.csv")








