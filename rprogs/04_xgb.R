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
library(Matrix)

################
# Prepare Data #
################

# Load Data #
combined_data <- readRDS("input/combined_data.rds")

# Create Sparse Matrices for Modeling #
train_pred_matrix_sparse <- combined_data %>%
  filter(data == "train") %>%
  select(-c(data, survived)) %>%
  as.matrix() %>%
  as("dgCMatrix")

train_class_matrix <-  combined_data %>%
  filter(data == "train") %>%
  select(survived) %>%
  as.matrix()

train_class_df <- train_class_matrix %>% as.data.frame()

val_pred_matrix_sparse <- combined_data %>%
  filter(data == "val") %>%
  select(-c(data, survived)) %>%
  as.matrix() %>%
  as("dgCMatrix")

val_class_matrix <-  combined_data %>%
  filter(data == "val") %>%
  select(survived) %>%
  as.matrix()

test_pred_matrix_sparse <- combined_data %>%
  filter(data == "test") %>%
  select(-c(data)) %>%
  as.matrix() %>%
  as("dgCMatrix")

# Create Model #
set.seed(25532523)

xgb_grid <- expand.grid(eta = seq(from = 0.1, to = 0.5, by = 0.05),
                        max_depth = seq(from = 1, to = 4, by = 1),
                        nround = seq(from = 50, to = 200, by = 5),
                        min_child_weight = 1/sqrt(length(train_class_df[train_class_df$survived == 1,])/nrow(train_class_df)),
                        colsample_bytree = seq(from = 0.2, to = 0.5, by = 0.05))

xgb_models <- list()
xgb_train_predictions <- list()
xgb_val_predictions <- list()
xgb_test_predictions <- list()

for (i in 1:nrow(xgb_grid)){
  
  set.seed(8451)
  
  xgb_model <- xgboost(data = train_pred_matrix_sparse,
                       label = train_class_matrix,
                       objective = "binary:logistic",
                       eta = xgb_grid$eta[i],
                       max_depth = xgb_grid$max_depth[i],
                       colsample_bytree = xgb_grid$colsample_bytree[i],
                       mid_child_weight = xgb_grid$min_child_weight[i],
                       nthread = 5,
                       nround = xgb_grid$nround[i],
                       verbose = 0)
  
  xgb_train_raw <- predict(xgb_model, train_pred_matrix_sparse)
  
  xgb_train_results_df <- data.frame(predictions = as.numeric(xgb_train_raw > 0.5),
                                     survived = as.character(train_class_matrix)) %>%
    mutate(correct = ifelse(predictions == survived, TRUE, FALSE))
  
  xgb_grid$train_correct_rate[i] <- nrow(xgb_train_results_df[xgb_train_results_df$correct == TRUE,])/nrow(xgb_train_results_df)
  
  xgb_val_raw <- predict(xgb_model, val_pred_matrix_sparse)
  
  xgb_val_results_df <- data.frame(predictions = as.numeric(xgb_val_raw > 0.5),
                                     survived = as.character(val_class_matrix)) %>%
    mutate(correct = ifelse(predictions == survived, TRUE, FALSE))
  
  xgb_grid$val_correct_rate[i] <- nrow(xgb_val_results_df[xgb_val_results_df$correct == TRUE,])/nrow(xgb_val_results_df)
  
  xgb_test_raw <- predict(xgb_model, test_pred_matrix_sparse)
  
  xgb_models[[i]] <- xgb_model
  xgb_train_predictions[[i]] <- as.numeric(xgb_train_raw > 0.5)
  xgb_val_predictions[[i]] <- as.numeric(xgb_val_raw > 0.5)
  xgb_test_predictions[[i]] <- as.numeric(xgb_test_raw > 0.5)
  
  if (i %% 100 == 0){
    print(paste(round((i/nrow(xgb_grid))*100, 2), "percent of models completed."))
  }
  
}

# Check Results #
paste0("The best train prediction rate is: ",
       round(xgb_grid$train_correct_rate[which.max(xgb_grid$val_correct_rate)], 2))

paste0("The best validation prediction rate is: ",
       round(xgb_grid$val_correct_rate[which.max(xgb_grid$val_correct_rate)], 2))

################
# Save Objects #
################

# Save Best Model #
saveRDS(xgb_models[[which.max(xgb_grid$val_correct_rate)]],
        file = "output/xgb_mod.rds")

# Save Predictons #
saveRDS(xgb_train_predictions[[which.max(xgb_grid$val_correct_rate)]],
        file = "output/xgb_train_preds.rds")
saveRDS(xgb_val_predictions[[which.max(xgb_grid$val_correct_rate)]],
        file = "output/xgb_val_preds.rds")
saveRDS(xgb_test_predictions[[which.max(xgb_grid$val_correct_rate)]],
        file = "output/xgb_test_preds.rds")








