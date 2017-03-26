#!/usr/bin/env Rscript

###################################
##   NAME: 01_titanic_survival.R ##
##   COMP: Kaggle Titanic Intro  ##
## AUTHOR: Mark Roepke           ##
##   DATE: 2016-12-10            ##
###################################

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
library(ranger)
library(xgboost)
library(Matrix)

################
# PREPARE DATA #
################

# LOAD TRAIN #
train <- read_csv("input/train.csv") %>%
# FORMAT VARIABLES #
         mutate(survived = Survived,
                first_class = ifelse(Pclass == 1, 1, 0),
                second_class = ifelse(Pclass == 2, 1, 0),
                third_class = ifelse(Pclass == 3, 1, 0),
                female = ifelse(Sex == "female", 1, 0),
                age = Age,
                young_child = ifelse(Age < 5, 1, 0),
                old_child = ifelse(Age > 12, ifelse(Age < 18, 1, 0), 0),
                young_female_child = young_child*female,
                young_male_child = ifelse(female == 0, ifelse(young_child == 1, 1, 0), 0),
                sibling_spouse = SibSp,
                parent_child = Parch,
                mother = ifelse(parent_child == 1, ifelse(female == 1, ifelse(age > 18, 1, 0), 0), 0),
                ticket_pc = ifelse(substr(Ticket, 1, 2) == "PC", 1, 0),
                ticket_soto = ifelse(substr(Ticket, 1, 4) == "SOTO", 1, 0),
                ticket_ston = ifelse(substr(Ticket, 1, 4) == "STON", 1, 0),
                ticket_c_a = ifelse(substr(Ticket, 1, 4) == "C.A.", 1, 0),
                ticket_ca = ifelse(substr(Ticket, 1, 2) == "CA", 1, 0),
                ticket_a = ifelse(substr(Ticket, 1, 1) == "A", 1, 0),
                ticket_3492 = ifelse(substr(Ticket, 1, 4) == "3493", 1, 0),
                ticket_3470 = ifelse(substr(Ticket, 1, 4) == "3470", 1, 0),
                ticket_1137 = ifelse(substr(Ticket, 1, 4) == "1137", 1, 0),
                fare = Fare,
                cabin_a = ifelse(substr(Cabin, 1, 1) == "A", 1, 0),
                cabin_b = ifelse(substr(Cabin, 1, 1) == "B", 1, 0),
                cabin_c = ifelse(substr(Cabin, 1, 1) == "C", 1, 0),
                cabin_d = ifelse(substr(Cabin, 1, 1) == "D", 1, 0),
                cabin_e = ifelse(substr(Cabin, 1, 1) == "E", 1, 0),
                cabin_f = ifelse(substr(Cabin, 1, 1) == "F", 1, 0),
                cabin_g = ifelse(substr(Cabin, 1, 1) == "G", 1, 0),
                cabin_t = ifelse(substr(Cabin, 1, 1) == "T", 1, 0),
                embark_cherbourg = ifelse(Embarked == "C", 1, 0),
                embark_queenstown = ifelse(Embarked == "Q", 1, 0)) %>%
# DROP UNWANTED VARIABLES FROM DATA.FRAME #
         select(-c(PassengerId, 
                   Survived, 
                   Pclass, 
                   Name, 
                   Sex, 
                   Age, 
                   SibSp, 
                   Parch, 
                   Ticket, 
                   Fare, 
                   Cabin, 
                   Embarked)) %>%
  as.matrix()

# REPLACE NAs WITH THE MEAN TO MINIMIZE PREDICTOR IMPACT #
for (i in 2:ncol(train)){
  train[is.na(train[,i]),i] <- mean(train[,i], na.rm = TRUE)
}

train <- as.data.frame(train) %>%
  select(-ticket_3492)

# MANUALLY STANDARDIZE EACH VARIABLE #
train[2:ncol(train)] <- lapply(train[2:ncol(train)], scale)

for (i in 2:ncol(train)){
  train[is.na(train[,i]),i] <- mean(train[,i], na.rm = TRUE)
}

##############
##############
## MODELING ##
##############
##############

#########
# LASSO #
#########

# PREPARE DATA #
classification_matrix <- train %>% 
                         select(survived) %>%
                         as.matrix()
predictor_matrix <- train %>%
                    select(-c(survived)) %>%
                    as.matrix()

# CREATE MODELS #
lasso_cv <- cv.glmnet(x = predictor_matrix, y = classification_matrix, family = "binomial", standardize = FALSE)
lasso_mod <- glmnet(x = predictor_matrix, y = classification_matrix, family = "binomial", standardize = FALSE, lambda = lasso_cv$lambda.min)

# CHECK RESULTS #
lasso_results_df <- data.frame(predictions = as.character(predict(lasso_mod, 
                                                                  s = lasso_cv$lambda.min, 
                                                                  newx = predictor_matrix,
                                                                  type = "class")), 
                               survived = train$survived) %>%
  mutate(correct = ifelse(predictions == survived, TRUE, FALSE))

nrow(lasso_results_df[lasso_results_df$correct == TRUE,])/nrow(lasso_results_df)

#################
# RANDOM FOREST #
#################

# Create Paramter Grid #

rf_grid <- expand.grid(num_trees = seq(from = 100, to = 300, length = 100),
                       m_try = seq(from = 11, to = 25, length = 15))

rf_models <- list()
rf_predictions <- list()

# Loop for Creating Forests #
set.seed(41071)

for (i in 1:nrow(rf_grid)){
  forest <- ranger(survived ~ ., data = train, num.trees = rf_grid$num_trees[i], 
                   mtry = rf_grid$m_try[i], write.forest = TRUE, 
                   classification = TRUE, importance = "impurity")

  results_df <- data.frame(predictions = forest$predictions, survived = train$survived) %>%
                  mutate(correct = ifelse(predictions == survived, TRUE, FALSE))

  rf_grid$correct_rate[i] <- nrow(results_df[results_df$correct == TRUE,])/nrow(results_df)
  
  rf_predictions[[i]] <- forest$predictions
  rf_models[[i]] <- forest

  if (i %% 100 == 0){
    print(paste(round((i/nrow(rf_grid))*100, 2), "percent of models completed."))
  }
}

# Cj #

print(paste0("The optimal parameters are mtry = ", 
             rf_grid$m_try[which.max(rf_grid$correct_rate)], 
             " and num.trees = ", 
             rf_grid$num_trees[which.max(rf_grid$correct_rate)]))

print(paste0("The best prediction rate is ",
             rf_grid$correct_rate[which.max(rf_grid$correct_rate)]))

# Save Best Model #
rf_model <- rf_models[[which.max(rf_grid$correct_rate)]]


#############################
# Extreme Gradient Boosting #
#############################

# Prepare Data #
pred_matrix_sparse <- as(predictor_matrix, "dgCMatrix")

# Create Model #
set.seed(41071)

xgb_grid <- expand.grid(eta = seq(from = 0.2, to = 0.25, by = 0.05),
                        max_depth = seq(from = 4, to = 8, by = 1),
                        nround = seq(from = 50, to = 90, by = 4),
                        min_child_weight = 1/sqrt(nrow(train[train$survived == 1,])/nrow(train)),
                        colsample_bytree = seq(from = 0.3, to = 0.4, by = 0.02))

xgb_models <- list()
xgb_predictions <- list()

for (i in 1:nrow(xgb_grid)){
  
  set.seed(8451)
  
  xgb_model <- xgboost(data = pred_matrix_sparse,
                       label = classification_matrix,
                       objective = "binary:logistic",
                       eta = xgb_grid$eta[i],
                       max_depth = xgb_grid$max_depth[i],
                       colsample_bytree = xgb_grid$colsample_bytree[i],
                       mid_child_weight = xgb_grid$min_child_weight[i],
                       nthread = 5,
                       nround = xgb_grid$nround[i],
                       verbose = 0)
 
  xgb_raw <- predict(xgb_model, pred_matrix_sparse)
  xgb_preds <- as.numeric(xgb_raw > 0.5)
  
  xgb_results_df <- data.frame(predictions = xgb_preds,
                               survived = train$survived) %>%
    mutate(correct = ifelse(predictions == survived, TRUE, FALSE))
  
  xgb_grid$correct_rate <- nrow(xgb_results_df[xgb_results_df$correct == TRUE,])/nrow(xgb_results_df)
  
  xgb_models[[i]] <- xgb_model
  xgb_predictions[[i]] <- xgb_preds
  
  if (i %% 100 == 0){
    print(paste(round((i/nrow(xgb_grid))*100, 2), "percent of models completed."))
  }
   
}

# Check Results #
paste0("The best prediction rate is: ",
       round(xgb_grid$correct_rate[which.max(xgb_grid$correct_rate)], 2))

# Save Best Model #
xgb_model <- xgb_models[[which.max(xgb_grid$correct_rate)]]
