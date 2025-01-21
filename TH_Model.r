### ADITYA KUMAR ROY ###
### MODEL IMPLEMENTATION, EVALUATION & OPTIMIZATION ###

## Loading Libraries
library(caret)
library(car)
library(dplyr)
library(nnet)
library(glmnet)
library(rpart)
library(randomForest)
library(Metrics)
library(e1071)
library(class)
library(tidyverse)
library(yardstick)
library(gmodels)
library(pROC)
library(xgboost)

## Loading Dataset
Weather_dataset_1_ <- read_csv("Weather_dataset-_1_.csv")
model <- Weather_dataset_1_[c(1:538),-c(7,8)]

## Data Pre-Processing
colnames(model) <- c("Date","Min_Temp","Max_Temp","Status","Avg_Temp","Wind")
model$Date <- dmy(model$Date)
model <- model %>% distinct() %>% drop_na()

## Train-Test Split
set.seed(123)
x <- model[,-1]
index <- createDataPartition(x$Status, p = 0.8, list = FALSE)
train <- x[index, ]
test <- x[-index, ]

## Model Implementation with Hyperparameter Tuning




# Load necessary libraries
library(readxl)
library(randomForest)
library(caret)
library(lubridate)



# Extract numeric features from the Date column
model$Year <- year(model$Date)
model$Month <- month(model$Date)
model$Day <- day(model$Date)

# Select features and target
features <- model[, c("Min_Temp","Max_Temp","Wind", "Year", "Month", "Day")]
target <- model$Avg_Temp

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)
trainData <- features[trainIndex, ]
trainTarget <- target[trainIndex]
testData <- features[-trainIndex, ]
testTarget <- target[-trainIndex]

# Create the model using randomForest
model <- randomForest(x = trainData, y = trainTarget, ntree = 100, random_state = 42)
model$rsq
# Make predictions
y_pred <- predict(model, testData)

# Evaluate the model
mae <- mean(abs(y_pred - testTarget))
mse <- mean((y_pred - testTarget)^2)
RMSE(y_pred,testTarget)



#plotting the model
ggplot()+
  geom_point(aes(x = testTarget,y = y_pred),col = "blue")+
  geom_abline(slope = 1,intercept = 0,col = "red")+
  labs(title = "Actual vs Predict ",x = "Actual",
       y = " Predicted",caption = "Random Forest Regressor")+
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5, color = "darkblue"), # Center and style title
    plot.subtitle = element_text(size = 14, hjust = 0.5, color = "gray30"), # Style subtitle
    plot.caption = element_text(size = 10, face = "italic", color = "darkgreen"), # Style caption
    axis.title = element_text(face = "bold"), # Bold axis titles
    panel.grid.major = element_line(color = "orange"), # Light grid lines
    panel.grid.minor = element_blank(),# Remove minor grid lines
    plot.background = element_rect(fill = "lightblue"),
    panel.background = element_rect(fill = "lightcoral")
  )








# Train a linear regression model
model_lr <- lm(trainTarget ~ ., data = trainData)
summary(model_lr)
# Make predictions on the test set
y_pred_lr <- predict(model_lr, newdata = testData)

# Calculate Mean Absolute Error (MAE)
mae_lr <- mean(abs(y_pred_lr - testTarget))
cat("MAE:", mae_lr, "\n")

# Calculate Mean Squared Error (MSE)
mse_lr <- mean((y_pred_lr - testTarget)^2)
cat("MSE:", mse_lr, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse_lr <- sqrt(mse_lr)
cat("RMSE:", rmse_lr, "\n")

ggplot()+
  geom_point(aes(x = testTarget,y = y_pred_lr),col = "blue")+
  geom_abline(slope = 1,intercept = 0,col = "red")+
  labs(title = "Actual vs Predict ",x = "Actual",
       y = " Predicted",caption = "Linear Regressor")+
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5, color = "darkblue"), # Center and style title
    plot.subtitle = element_text(size = 14, hjust = 0.5, color = "gray30"), # Style subtitle
    plot.caption = element_text(size = 10, face = "italic", color = "darkgreen"), # Style caption
    axis.title = element_text(face = "bold"), # Bold axis titles
    panel.grid.major = element_line(color = "orange"), # Light grid lines
    panel.grid.minor = element_blank(),# Remove minor grid lines
    plot.background = element_rect(fill = "lightblue"),
    panel.background = element_rect(fill = "lemonchiffon4")
  )



