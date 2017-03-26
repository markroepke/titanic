#!/usr/bin/env Rscript

####################################
##    NAME: 02_lasso_reg.R        ##
##    COMP: Kaggle Titanic Intro  ##
##  AUTHOR: Mark Roepke           ##
##    DATE: 2016-12-10            ##
## PURPOSE: Create LASSO model    ##
####################################

#########
# SETUP #
#########

# PREPARE SCRIPT #
rm(list = ls())
#setwd("/mnt/c/Users/roepk_000/OneDrive/Documents/Data/data/kaggle/titanic")
setwd("C://Users//roepk_000//OneDrive//Documents//Data//data//kaggle//titanic")

# LOAD PACKAGES #
library(tidyverse)
library(glmnet)

################
# Prepare Data #
################

# Load data #
combined_data <- readRDS("input/combined_data.rds")

# Create matrices #
class_train_matrix <- combined_data %>%
  filter(data == "train") %>%
  select(survived) %>%
  as.matrix()

pred_train_matrix <- combined_data %>%
  filter(data == "train") %>%
  select(-c(survived, data)) %>%
  as.matrix()

class_val_matrix <- combined_data %>%
  filter(data == "val") %>%
  select(survived) %>%
  as.matrix()

pred_val_matrix <- combined_data %>%
  filter(data == "val") %>%
  select(-c(survived, data)) %>%
  as.matrix()

pred_test_matrix <- combined_data %>%
  filter(data == "test") %>%
  select(-c(survived, data)) %>%
  as.matrix()

############
# Modeling #
############

# Crossvalidate to find optimal lambda #
lasso_cv <- cv.glmnet(x = pred_train_matrix, y = class_train_matrix, family = "binomial", standardize = FALSE)

# Create model using optimal lambda
lasso_mod <- glmnet(x = pred_train_matrix, y = class_train_matrix, family = "binomial", standardize = FALSE, lambda = lasso_cv$lambda.min)

# Check training results #
lasso_results_df <- data.frame(predictions = as.character(predict(lasso_mod, 
                                                                  s = lasso_cv$lambda.min, 
                                                                  newx = pred_train_matrix,
                                                                  type = "class")), 
                               survived = filter(combined_data,
                                                 data == "train")$survived) %>%
  mutate(correct = ifelse(predictions == survived, TRUE, FALSE))

(train_lasso_rate <- nrow(lasso_results_df[lasso_results_df$correct == TRUE,])/nrow(lasso_results_df))

# Check val results #
lasso_results_df <- data.frame(predictions = as.character(predict(lasso_mod, 
                                                                  s = lasso_cv$lambda.min, 
                                                                  newx = pred_val_matrix,
                                                                  type = "class")), 
                               survived = filter(combined_data,
                                                 data == "val")$survived) %>%
  mutate(correct = ifelse(predictions == survived, TRUE, FALSE))

(val_lasso_rate <- nrow(lasso_results_df[lasso_results_df$correct == TRUE,])/nrow(lasso_results_df))

################
# Save Objects #
################

# Save model info #
saveRDS(lasso_mod, 
        file = "output/lasso_mod.rds")
saveRDS(lasso_cv$lambda.min, 
        file = "output/lasso_best_lam.rds")

# Save training predictions #
saveRDS(as.double(predict(lasso_mod, 
                          s = lasso_cv$lambda.min, 
                          newx = pred_train_matrix,
                          type = "class")),
        file = "output/lasso_train_preds.rds")
saveRDS(as.double(predict(lasso_mod, 
                          s = lasso_cv$lambda.min, 
                          newx = pred_val_matrix,
                          type = "class")),
        file = "output/lasso_val_preds.rds")
saveRDS(as.double(predict(lasso_mod, 
                          s = lasso_cv$lambda.min, 
                          newx = pred_test_matrix,
                          type = "class")),
        file = "output/lasso_test_preds.rds")




