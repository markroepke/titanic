#!/usr/bin/env Rscript

#####################################
##    NAME: 03_random_forests.R    ##
##    COMP: Kaggle Titanic Intro   ##
##  AUTHOR: Mark Roepke            ##
##    DATE: 2016-12-10             ##
## PURPOSE: Create RF models       ##
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
library(ranger)

################
# Prepare Data #
################

# Load data #
combined_data <- readRDS("input/combined_data.rds")

# Create data frames for modeling #
train_data <- combined_data %>%
  filter(data == "train") %>%
  select(-data)

val_data <- combined_data %>%
  filter(data == "val") %>%
  select(-data)

test_data <- combined_data %>%
  filter(data == "test") %>%
  select(-c(data))

############
# Modeling #
############

# Create Paramter Grid #

rf_grid <- expand.grid(num_trees = seq(from = 100, to = 200, by = 1),
                       m_try = seq(from = 11, to = 25, by = 1))

rf_models <- list()
rf_train_predictions <- list()
rf_val_predictions <- list()
rf_test_predictions <- list()

# Loop for Creating Forests #
set.seed(41071)

for (i in 1:nrow(rf_grid)){
  forest <- ranger(survived ~ ., data = train_data, num.trees = rf_grid$num_trees[i], 
                   mtry = rf_grid$m_try[i], write.forest = TRUE, 
                   classification = TRUE, importance = "impurity")
  
  train_results_df <- data.frame(predictions = forest$predictions,
                                 survived = train_data$survived) %>%
    mutate(correct = ifelse(predictions == survived, TRUE, FALSE))
  
  val_results_df <- data.frame(predictions = predict(forest, val_data)$predictions,
                               survived = val_data$survived) %>%
    mutate(correct = ifelse(predictions == survived, TRUE, FALSE))
  
  rf_grid$train_correct_rate[i] <- nrow(train_results_df[train_results_df$correct == TRUE,])/nrow(train_results_df)
  rf_grid$val_correct_rate[i] <- nrow(val_results_df[val_results_df$correct == TRUE,])/nrow(val_results_df)
  
  rf_train_predictions[[i]] <- forest$predictions
  rf_val_predictions[[i]] <- predict(forest, val_data)$predictions
  rf_test_predictions[[i]] <- predict(forest, test_data)$predictions
    
  rf_models[[i]] <- forest
  
  if (i %% 100 == 0){
    print(paste(round((i/nrow(rf_grid))*100, 2), "percent of models completed."))
  }
}

# Find optimal parameters #

print(paste0("The optimal parameters are mtry = ", 
             rf_grid$m_try[which.max(rf_grid$val_correct_rate)], 
             " and num.trees = ", 
             rf_grid$num_trees[which.max(rf_grid$val_correct_rate)]))

print(paste0("The optimal train prediction rate is ",
             round(rf_grid$train_correct_rate[which.max(rf_grid$val_correct_rate)], 2)))

print(paste0("The optimal validation prediction rate is ",
             round(rf_grid$val_correct_rate[which.max(rf_grid$val_correct_rate)], 2)))

################
# Save Objects #
################

# Save Best Model #
saveRDS(rf_models[[which.max(rf_grid$val_correct_rate)]],
        file = "output/rf_mod.rds")

# Save Predictons #
saveRDS(rf_train_predictions[[which.max(rf_grid$val_correct_rate)]],
        file = "output/rf_train_preds.rds")
saveRDS(rf_val_predictions[[which.max(rf_grid$val_correct_rate)]],
        file = "output/rf_val_preds.rds")
saveRDS(rf_test_predictions[[which.max(rf_grid$val_correct_rate)]],
        file = "output/rf_test_preds.rds")

