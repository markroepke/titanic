#!/usr/bin/env Rscript

#####################################
##    NAME: 05_ensemble.R          ##
##    COMP: Kaggle Titanic Intro   ##
##  AUTHOR: Mark Roepke            ##
##    DATE: 2016-12-10             ##
## PURPOSE: Create ensemble model  ##
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
library(glmnet)

################
# Prepare Data #
################

# Read in all data #
combined_data <- readRDS("input/combined_data.rds")

lasso_train_preds <- readRDS("output/lasso_train_preds.rds")
lasso_val_preds <- readRDS("output/lasso_val_preds.rds")
lasso_test_preds <- readRDS("output/lasso_test_preds.rds")

rf_train_preds <- readRDS("output/rf_train_preds.rds")
rf_val_preds <- readRDS("output/rf_val_preds.rds")
rf_test_preds <- readRDS("output/rf_test_preds.rds")

xgb_train_preds <- readRDS("output/xgb_train_preds.rds")
xgb_val_preds <- readRDS("output/xgb_val_preds.rds")
xgb_test_preds <- readRDS("output/xgb_test_preds.rds")

# Combine predictor data #
total_train_preds <- data.frame(filter(combined_data, data == "train"),
                                lasso_preds = lasso_train_preds, 
                                rf_preds = rf_train_preds,
                                xgb_preds = xgb_train_preds) %>%
  select(-c(survived, data)) %>%
  as.matrix()

total_val_preds <- data.frame(filter(combined_data, data == "val"),
                              lasso_preds = lasso_val_preds, 
                              rf_preds = rf_val_preds,
                              xgb_preds = xgb_val_preds) %>%
  select(-c(survived, data)) %>%
  as.matrix()

total_test_preds <- data.frame(filter(combined_data, data == "test"),
                               lasso_preds = lasso_test_preds, 
                               rf_preds = rf_test_preds,
                               xgb_preds = xgb_test_preds) %>%
  select(-c(survived, data)) %>%
  as.matrix()

# Create classification matrices #
class_train_matrix <- combined_data %>%
  filter(data == "train") %>%
  select(survived) %>%
  as.matrix()

class_val_matrix <- combined_data %>%
  filter(data == "val") %>%
  select(survived) %>%
  as.matrix()

###########################
# Create Stacked Ensemble #
###########################

# Crossvalidate to find optimal lambda #
ens_cv <- cv.glmnet(x = total_train_preds, y = class_train_matrix, family = "binomial", standardize = FALSE)

# Create model using optimal lambda
ens_mod <- glmnet(x = total_train_preds, y = class_train_matrix, family = "binomial", standardize = FALSE, lambda = ens_cv$lambda.min)

# Check training results #
ens_results_df <- data.frame(predictions = as.character(predict(ens_mod, 
                                                                s = ens_cv$lambda.min, 
                                                                newx = total_train_preds,
                                                                type = "class")), 
                             survived = filter(combined_data,
                                               data == "train")$survived) %>%
  mutate(correct = ifelse(predictions == survived, TRUE, FALSE))

(train_lasso_rate <- nrow(ens_results_df[ens_results_df$correct == TRUE,])/nrow(ens_results_df))

# Check val results #
ens_results_df <- data.frame(predictions = as.character(predict(ens_mod, 
                                                                s = ens_cv$lambda.min, 
                                                                newx = total_val_preds,
                                                                type = "class")), 
                             survived = filter(combined_data,
                                               data == "val")$survived) %>%
  mutate(correct = ifelse(predictions == survived, TRUE, FALSE))

(val_lasso_rate <- nrow(ens_results_df[ens_results_df$correct == TRUE,])/nrow(ens_results_df))

















