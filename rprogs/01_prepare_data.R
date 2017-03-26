#!/usr/bin/env Rscript

####################################
##    NAME: 01_prepare_data.R     ##
##    COMP: Kaggle Titanic Intro  ##
##  AUTHOR: Mark Roepke           ##
##    DATE: 2016-12-10            ##
## PURPOSE: Import and prepare    ##
##          date for modeling     ##
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

# Load train data #
set.seed(8451)
train <- read_csv("input/train.csv") %>%
  mutate(runif = runif(n = nrow(.),
                       min = 0,
                       max = 1),
         data = ifelse(runif <= 0.8,
                       "train",
                       "val")) %>%
  select(-runif)

# Load test data #
test <- read_csv("input/test.csv") %>%
  mutate(data = "test")

# Combine data #
combined_data <- train %>%
  bind_rows(test) %>%
# Format variables #
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
            Embarked))

# Save train/test variable for reference
data_set_type <- combined_data$data

# Remove train/test variable from data set and convert to matrix
combined_data <- combined_data %>%
  select(-data) %>%
  as.matrix()

# Replace NAs with the mean to minimize missing data impact on predictors #
for (i in 2:ncol(combined_data)){
  combined_data[is.na(combined_data[,i]),i] <- mean(combined_data[,i], na.rm = TRUE)
}

# Convert back to data frame #
combined_data <- combined_data %>%
  as.data.frame()

# Manually standardize predictor variables #
combined_data[2:ncol(combined_data)] <- lapply(combined_data[2:ncol(combined_data)], scale)
combined_data[2:ncol(combined_data)] <- lapply(combined_data[2:ncol(combined_data)], as.double)

# Replace new NAs with the mean to minimize missing data impact on predictors #
for (i in 2:ncol(combined_data)){
  combined_data[is.na(combined_data[,i]),i] <- mean(combined_data[,i], na.rm = TRUE)
}

# Add train/type variable back to dataset #
combined_data$data <- data_set_type

###############
# Saving Data #
###############

saveRDS(combined_data, file = "input/combined_data.rds")