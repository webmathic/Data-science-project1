library('tidyverse')
library('mice')
library('ISLR')
library('tidyr')
library('dplyr')
library('corrplot')
library("faraway")
library("olsrr")
library("e1071")
library("caret")
library(rpart)
library(rpart.plot)
library(readr)

house_data <- read.csv('house-data.csv', header=T)
head(house_data)

#checking the total number of values in each column
missing.values <- house_data %>%
  gather(key = "key", value = "val") %>%
  mutate(is.missing = is.na(val)) %>%
  group_by(key, is.missing) %>%
  summarise(num.missing = n()) %>%
  filter(is.missing==T) %>%
  select(-is.missing) %>%
  arrange(desc(num.missing))
View(missing.values)

#viewing the missing data graphically
missing.values %>%
  ggplot() +
  geom_bar(aes(x=key, y=num.missing),  stat = 'identity') +
  labs(x='variable', y="number of missing values", 
       title='Number of missing values') 

house_data <- house_data[, !names(house_data) %in% c("PoolQC", "MiscFeature", "Alley", "Fence")]
view(house_data)

# Delete the with a lot of missing values
house_data <- house_data[, !names(house_data) %in% c("Id","PoolQC", "MiscFeature", "Alley", "Fence")]
view(house_data)

# data types of the data set
str(house_data)

#mean imputation of Access to electricity (\% of population)
mean <- mean(house_data$LotFrontage, na.rm = TRUE)

#replacing n/a with the mean 
house_data[is.na(house_data$LotFrontage), "LotFrontage"] <- mean

#mean imputation of Access to electricity (\% of population)
mean2 <- mean(house_data$MasVnrArea, na.rm = TRUE)

#replacing n/a with the mean 
house_data[is.na(house_data$MasVnrArea), "MasVnrArea"] <- mean2

#rechecking the total number of values in each column
missing.values <- house_data %>%
  gather(key = "key", value = "val") %>%
  mutate(is.missing = is.na(val)) %>%
  group_by(key, is.missing) %>%
  summarise(num.missing = n()) %>%
  filter(is.missing==T) %>%
  select(-is.missing) %>%
  arrange(desc(num.missing))
View(missing.values)

#summary of the data set
summary(house_data)


# Convert categorical variables to numerical values
house_data$Street <- as.numeric(factor(house_data$Street))

house_data$Utilities <- as.numeric(factor(house_data$Utilities))

house_data$LotConfig <- as.numeric(factor(house_data$LotConfig))

house_data$Neighborhood <- as.numeric(factor(house_data$Neighborhood))

house_data$Condition1 <- as.numeric(factor(house_data$Condition1))

house_data$Condition2 <- as.numeric(factor(house_data$Condition2))

house_data$BldgType <- as.numeric(factor(house_data$BldgType))

house_data$HouseStyle <- as.numeric(factor(house_data$HouseStyle))

house_data$RoofStyle <- as.numeric(factor(house_data$RoofStyle))

house_data$RoofMatl <- as.numeric(factor(house_data$RoofMatl))

house_data$Exterior1st <- as.numeric(factor(house_data$Exterior1st))

house_data$ExterQual <- as.numeric(factor(house_data$ExterQual))

house_data$ExterCond <- as.numeric(factor(house_data$ExterCond))


house_data$BsmtQual <- as.numeric(factor(house_data$BsmtQual))

house_data$BsmtCond <- as.numeric(factor(house_data$BsmtCond))

house_data$Heating <- as.numeric(factor(house_data$Heating))

house_data$KitchenQual <- as.numeric(factor(house_data$KitchenQual))

house_data$Functional <- as.numeric(factor(house_data$Functional))

house_data$GarageType <- as.numeric(factor(house_data$GarageType))

house_data$GarageCond <- as.numeric(factor(house_data$GarageCond))

house_data$PavedDrive <- as.numeric(factor(house_data$PavedDrive))

house_data$SaleType <- as.numeric(factor(house_data$SaleType))

house_data$SaleCondition <- as.numeric(factor(house_data$SaleCondition))

unique(house_data[c("OverallCond")])
# Divide houses based on overall condition
house_data$OverallCond <- cut(house_data$OverallCond, breaks = c(1, 3, 6, 10),
                            labels = c("Poor", "Average", "Good"))
head(house_data$OverallCond)
view(house_data)

# split dataset
set.seed(999) 
train.index = sample(x=1460, size=1160)
house_data.train = house_data[train.index, ]
house_data.test = house_data[-train.index, ]

# Remove rows with missing values from the test data
# Remove rows with missing values from the test data
house_data.train <- house_data.train[complete.cases(house_data.train),]

house_data.test <- house_data.test[complete.cases(house_data.test),]
house_data <- house_data[complete.cases(house_data),]


# Multinomial logistic regression 
LR <- nnet::multinom(OverallCond ~ ., data = house_data.train, family = "binomial")

summary(LR)



# Make predictions
predicted.classes <- predict(LR, house_data.test)
head(predicted.classes)

# confusion matrix
table(predicted.classes, house_data.test$OverallCond)

# Model accuracy
mean(predicted.classes == house_data.test$OverallCond)


# Train an SVM classifier
svm_model <- svm(OverallCond ~ ., data = house_data.train, kernel = "linear", cost = 10,
                 scale = T)
summary(svm_model)

print(svm_model)


# Make predictions on the test data
svm_pred <- predict(svm_model, newdata = house_data.test)
head(svm_pred)

# confusion matrix
table(svm_pred, house_data.test$OverallCond)

# Evaluate the model accuracy
mean(svm_pred == house_data.test$OverallCond)

# Load the required library
library(randomForest)

# Train a random forest classifier
rf_models <- randomForest(OverallCond ~ ., data = house_data.train, ntree = 500, mtry = 2, 
                         cv.fold = 10)

print(rf_models)

# Make predictions on the test data
rf_pred <- predict(rf_models, newdata = house_data.test)

# confusion matrix
table(rf_pred, house_data.test$OverallCond)

# Evaluate the model accuracy
mean(rf_pred == house_data.test$OverallCond)

# Calculate the misclassification error
mean(rf_pred != house_data.test$OverallCond)


require(tree)
tree.house <- tree(OverallCond ~ ., data= house_data.train)
tree.house
summary(tree.house)



# Make predictions on the test data
tree_pred <- predict(tree.house, newdata = house_data.test, type = "class")

# confusion matrix
table(tree_pred, house_data.test$OverallCond)

# Evaluate the model accuracy
mean(tree_pred == house_data.test$OverallCond)

plot(tree.house); text(tree.house)

# Use the prune.misclass() function to prune the tree to the best seven terminal nodes
(pruned.tree <- prune.misclass(tree.house, best=7))
plot(pruned.tree); text(tree.house)

var_impor <- data.frame(Variable = row.names(importance(rf_models)), Importance = importance(rf_models)[,1])
var_impor

# Create plot of variable importance measures using bar plot
ggplot(var_impor, aes(x = reorder(Variable, -Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#0072B2") +
  labs(title = "Predictor Importance Plot", x = "Predictor", y = "Importance") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1))

# fit a decision tree regression model to the training data
tree_model <- rpart(SalePrice ~ ., data = house_data.train)

# plot the decision tree
rpart.plot(tree_model, type = 4)

# make predictions on the test data
tree.preds <- predict(tree_model, newdata = house_data.test)
summary(tree.preds)

# tree model performance
r2 <- 1 - sum((tree.preds - house_data.test$SalePrice)^2) / sum((house_data.test$SalePrice - mean(house_data.test$SalePrice))^2)
cat("R-squared value:", r2, "\n")

# Calculate the misclassification error
mean(tree.preds != house_data.test$SalePrice)
rmse <- sqrt(mean((tree.preds - house_data.test$SalePrice)^2))
cat("RMSE:", rmse, "\n")

# train the SVR model
svm_models <- svm(SalePrice ~ ., data = house_data.train, kernel = "linear", cost = 100, gamma = 0.1)

# make predictions on the test set
svr.preds <- predict(svm_models, house_data.test)

# calculate the mean squared error and root mean squared error
mse <- mean((svr.preds - house_data.test$SalePrice)^2)
rmse <- sqrt(mse)

cat("Mean squared error:", mse, "\n")
cat("Root mean squared error:", rmse, "\n")

# calculate the R-squared value
r2.svr <- 1 - sum((svr.preds - house_data.test$SalePrice)^2) / sum((house_data.test$SalePrice - mean(house_data.test$SalePrice))^2)
cat("R-squared value:", r2.svr, "\n")




# Train a linear regression model
linear_model <- lm(SalePrice ~ ., data = house_data.train)


# Set up the training control for cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the linear regression model with cross-validation
lm_model <- train(SalePrice ~ ., data = house_data.train, method = "lm",
                  trControl = train_control)

# Print the cross-validation results
print(lm_model)



# Define the input features and output variable
X <-  house_data %>% select(-SalePrice)
y <- house_data$SalePrice

set.seed(123)
cv <- trainControl(method = "cv", number = 10)
tree_model <- train(X, y, method = "rpart", trControl = cv)
cv_error <- tree_model$results$RMSE[1]

# Print the cross-validation error
cat("Cross-validation error:", cv_error, "\n")

library(boot)

# Define the function to compute the error
boot_error <- function(data, indices) {
  x <- data[indices, -ncol(data)]
  Y <- data$SalePrice[indices]
  preds <- predict(tree_model, newdata = x)
  error <- sqrt(mean((preds - Y)^2))
  return(error)
}

# Specify the number of bootstrap samples
num_bootstraps <- 1000

# Compute the bootstrap error using the boot function
set.seed(123)
boot_results <- boot(data = house_data, statistic = boot_error, R = num_bootstraps)

# Compute the mean and standard error of the bootstrap error
boot_error_mean <- mean(boot_results$t)
boot_error_se <- sd(boot_results$t)

# Print the results
cat("Bootstrap error mean:", boot_error_mean, "\n")
cat("Bootstrap error standard error:", boot_error_se, "\n")


# Train the model on the training set using the rpart algorithm
model <- train(SalePrice ~ ., data = house_data.train, method = "rpart")

# Make predictions on the test set
predictions <- predict(model, newdata = house_data.test)

# Calculate the test error (root mean squared error in this case)
test_error <- RMSE(predictions, house_data.test$SalePrice)

# Print the test error
cat("Test error:", test_error, "\n")

# it is also good to investigate the collinearity among the prediction in the data set.
# Select only numeric variables from house_data
numeric_house_data <- house_data %>% 
  select_if(is.numeric)

# Calculate correlation matrix for numeric_house_data
correlation_matrix <- cor(numeric_house_data)

# Print correlation matrix
print(correlation_matrix)


# Find the most highly correlated variables
correlation_pairs <- as.data.frame(as.table(correlation_matrix)) %>%
  filter(Freq != 1 & Freq != -1) %>%    # remove variables with perfect correlation
  arrange(desc(abs(Freq))) %>%         # sort by absolute value of correlation coefficient
  head(n = 20)                         # select top 20 pairs

# Print the most highly correlated variables
print(correlation_pairs)

var_imp <- varImp(tree_model$finalModel)
print(var_imp)



# Train the random forest model
rf_model <- randomForest(SalePrice ~ ., data = house_data.train, ntree = 500, mtry = 4)

# Make predictions on the test set
rf_preds <- predict(rf_model, newdata = house_data.test)

# Evaluate the model using RMSE
rmse <- RMSE(rf_preds, house_data.test$SalePrice)
print(paste("Random Forest RMSE:", rmse))


# Define the training control
train_control <- trainControl(method = "cv", number = 10)

# Train the model using random forest
rf_model <- train(SalePrice ~ ., data = house_data.train, method = "rf", trControl = train_control)

# Print the mean squared error for the cross-validation
print(rf_model$results$RMSE)

#print the model performance
print(rf_model$results$Rsquared)



# Set the number of bootstrap samples
num_bootstraps <- 50

# Set up the bootstrap control
boot_control <- trainControl(method = "boot", number = num_bootstraps)

# Train the model using random forest
rf_model <- train(SalePrice ~ ., data = house_data.train, method = "rf", trControl = boot_control)

# Print the  variable importance measures
print(varImp(rf_model$finalModel))

# Train the model on the training set using the random forest algorithm
model <- train(SalePrice ~ ., data = house_data.train, method = "rf")

# Make predictions on the test set
predictions <- predict(model, newdata = house_data.test)

# Calculate the test error (root mean squared error in this case)
test_error <- RMSE(predictions, house_data.test$SalePrice)

# Print the test error
cat("Test error:", test_error, "\n")

#print the important variable
var_imp <- varImp(rf_model)
print(var_imp)

#plot the importance variable
plot(var_imp)

# Create plot of variable importance measures using bar plot
ggplot(var_imp, aes(x = reorder(Variable, -Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#0072B2") +
  labs(title = "Predictor Importance Plot", x = "Predictor", y = "Importance") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1))


# Create plot of variable importance measures using scatter plot
ggplot(var_imp, aes(x = reorder(Variable, -Importance), y = Importance)) +
  geom_point(stat = "identity", color = "blue") + # Change the color here
  labs(title = "Predictors Importance Plot", x = "Predictors", y = "Importance") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1))

