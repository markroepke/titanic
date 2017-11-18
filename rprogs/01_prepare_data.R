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
library(data.table)

#############
# Load Data #
#############

# Load training data
train <- fread("input/train.csv") %>%
  dplyr::mutate(data = "train")

# Load holdout data
test <- fread("input/test.csv") %>%
  dplyr::mutate(data = "test")

# Combine data
combined_data <- train %>%
  dplyr::bind_rows(test)

#######################
# Feature Engineering #
#######################

# Make all column names lower case for ease
names(combined_data) <- str_to_lower(names(combined_data))

# Add some informative columns to hint algorithm
combined_data <- combined_data %>%
  dplyr::mutate(child = ifelse(age < 18, 1, 0),
         young_child = ifelse(age < 5, 1, 0),
         middle_child = ifelse(between(age, 5, 12), 1, 0),
         old_child = ifelse(between(age, 13, 18), 1, 0),
         young_female_child = ifelse(sex == "female", ifelse(young_child == 1, 1, 0), 0),
         middle_female_child = ifelse(sex == "female", ifelse(middle_child == 1, 1, 0), 0),
         old_female_child = ifelse(sex == "female", ifelse(old_child == 1, 1, 0), 0),
         young_male_child = ifelse(sex == "male", ifelse(young_child == 1, 1, 0), 0),
         middle_male_child = ifelse(sex == "male", ifelse(middle_child == 1, 1, 0), 0),
         old_male_child = ifelse(sex == "male", ifelse(old_child == 1, 1, 0), 0),
         female_adult = ifelse(sex == "female", ifelse(child == 0, 1, 0), 0),
         mother = ifelse(parch == 1, ifelse(female_adult == 1, 1, 0), 0),
         wife = ifelse(female_adult == 1, ifelse(sibsp == 1, 1, 0), 0),
         male_adult = ifelse(sex == "male", ifelse(child == 0, 1, 0), 0),
         father = ifelse(parch == 1, ifelse(male_adult == 1, 1, 0), 0),
         husband = ifelse(male_adult == 1, ifelse(sibsp == 1, 1, 0), 0),
         family_size = sibsp + parch + 1,
         ticket_pc = ifelse(str_sub(ticket, 1, 2) == "PC", 1, 0),
         ticket_soto = ifelse(str_sub(ticket, 1, 4) == "SOTO", 1, 0),
         ticket_ston = ifelse(str_sub(ticket, 1, 4) == "STON", 1, 0),
         ticket_c_a = ifelse(str_sub(ticket, 1, 4) == "C.A.", 1, 0),
         ticket_ca = ifelse(str_sub(ticket, 1, 2) == "CA", 1, 0),
         ticket_a = ifelse(str_sub(ticket, 1, 1) == "A", 1, 0),
         ticket_3470 = ifelse(str_sub(ticket, 1, 4) == "3470", 1, 0),
         ticket_1137 = ifelse(str_sub(ticket, 1, 4) == "1137", 1, 0),
         cabin_a = ifelse(str_sub(cabin, 1, 1) == "A", 1, 0),
         cabin_b = ifelse(str_sub(cabin, 1, 1) == "B", 1, 0),
         cabin_c = ifelse(str_sub(cabin, 1, 1) == "C", 1, 0),
         cabin_d = ifelse(str_sub(cabin, 1, 1) == "D", 1, 0),
         cabin_e = ifelse(str_sub(cabin, 1, 1) == "E", 1, 0),
         cabin_f = ifelse(str_sub(cabin, 1, 1) == "F", 1, 0),
         cabin_g = ifelse(str_sub(cabin, 1, 1) == "G", 1, 0),
         cabin_t = ifelse(str_sub(cabin, 1, 1) == "T", 1, 0)
  )

# Family Groups
family_name <- sapply(combined_data$name, FUN = function(x) {str_split(x, pattern = "[,.]")[[1]][1]})
combined_data$family_group <- str_c(family_name, "-", combined_data$family_size)
combined_data$family_group <- ifelse(as.numeric(str_sub(combined_data$family_group, str_length(combined_data$family_group), str_length(combined_data$family_group))) < 3, NA, combined_data$family_group)

# Average fare by family group
combined_data <- combined_data %>%
  dplyr::filter(!is.na(family_group)) %>%
  dplyr::group_by(family_group) %>%
  dplyr::summarize(mean_fare = mean(fare)) %>%
  dplyr::right_join(combined_data, by = "family_group") %>%
  dplyr::ungroup()
  

# Titles
combined_data$title <- sapply(combined_data$name, FUN = function(x) {str_split(x, pattern = "[,.]")[[1]][2]})
combined_data$title <- str_replace_all(combined_data$title, " ", "")
combined_data$title[combined_data$title %in% c("Ms", "Miss")] <- "Miss"
combined_data$title[combined_data$title %in% c("Mme", "Mlle")] <- "Mlle"
combined_data$title[combined_data$title %in% c("Capt", "Don", "Major", "Sir")] <- "Sir"
combined_data$title[combined_data$title %in% c("Dona", "Lady", "theCountess", "Jonkheer")] <- "Lady"

# Remove excess variables
combined_data <- combined_data %>%
  select(-c(name))


###############
# Saving Data #
###############

write_rds(combined_data, path = "input/combined_data.rds")
