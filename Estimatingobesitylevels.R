# Loading required libraries
lapply(c("tidyverse", "caret", "nnet", "rpart", "rpart.plot", "randomForest", "ggplot2", "GGally", "pROC", 
         "reshape2", "ggcorrplot", "dplyr", "lime", "tidyr", "fastshap", "ggforce", "MLmetrics"), require, character.only = TRUE)

# Step 1: Loading and renaming columns for clarity
data <- read.csv("ObesityData.csv", stringsAsFactors = FALSE)

data <- data %>%
  rename(
    FamilyOverWeightHistory = family_history_with_overweight,HighCalorieFood = FAVC, VegIntakeFreq = FCVC,MainMealsPerDay = NCP,
    Snacking = CAEC,WaterIntake = CH2O,CalorieMonitoring = SCC,PhysicalActivity = FAF,ScreenTime = TUE,AlcoholConsumption = CALC,
    TransportMode = MTRANS,ObesityLevel = NObeyesdad
  )

# Step 2: Inspecting raw data
cat("\nColumn names:\n")
print(colnames(data))

cat("\nSample data (head):\n")
print(head(data))

cat("\nSummary of raw data:\n")
print(summary(data))

cat("\nMissing values per column:\n")
print(colSums(is.na(data)))

# Step 3: Exploratory data analysis
cat("\n Multi Class Target Distribution:\n")
print(table(data$ObesityLevel))

# 1. EDA - Class distribution Plot
ggplot(data, aes(x = ObesityLevel, fill = ObesityLevel)) +
  geom_bar() +
  labs(title = "Class Distribution of Obesity Levels",
       x = "Obesity Category", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set3")

# Defining variable types
numeric_cols <- c("Age", "Height", "Weight", "VegIntakeFreq", "MainMealsPerDay",
                  "WaterIntake", "PhysicalActivity", "ScreenTime")

categorical_cols <- c("Gender", "FamilyOverWeightHistory", "HighCalorieFood", "Snacking", 
                      "SMOKE", "CalorieMonitoring", "AlcoholConsumption", "TransportMode")

# 2. Box-plots of numeric features
data_long <- data %>%
  select(all_of(numeric_cols), ObesityLevel) %>%
  pivot_longer(cols = -ObesityLevel, names_to = "Feature", values_to = "Value")

ggplot(data_long, aes(x = ObesityLevel, y = Value, fill = ObesityLevel)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free_y") +
  labs(title = "Distribution of Numeric Features by Obesity Level",
       x = "Obesity Category", y = "Raw Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold"))

# 3. Correlation matrix of numeric features
cor_matrix <- cor(data[, numeric_cols])

ggcorrplot::ggcorrplot(cor_matrix, lab = TRUE, lab_size = 3,
                       title = "Correlation Matrix of Numeric Features",
                       type = "lower", colors = c("red", "white", "blue"))

# 4. Categorical vs Obesity Level
cat_data_long <- data %>%
  select(all_of(categorical_cols), ObesityLevel) %>%
  pivot_longer(cols = -ObesityLevel, names_to = "Feature", values_to = "Category")

ggplot(cat_data_long, aes(x = Category, fill = ObesityLevel)) +
  geom_bar(position = "fill") +
  facet_wrap(~ Feature, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Obesity Level vs. Categorical Features",
       y = "Proportion", x = "Category", fill = "Obesity Level") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Step 4: Preprocessing
data$ObesityLevel <- as.factor(data$ObesityLevel)
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)

# Normalizing numeric features
data[numeric_cols] <- scale(data[numeric_cols])

cat("\nSummary of scaled numeric variables:\n")
print(summary(data[numeric_cols]))

cat("\nSummary of factor variables:\n")
print(summary(data[categorical_cols]))

cat("\nFinal structure of the dataset:\n")
print(str(data))

# Step 5: One-Hot Encoding
dummies_model <- dummyVars(ObesityLevel ~ ., data = data, fullRank = TRUE)
encoded_data <- as.data.frame(predict(dummies_model, newdata = data))
model_data <- cbind(encoded_data, ObesityLevel = data$ObesityLevel)

cat("\nStructure of encoded data:\n")
print(str(model_data))

print(summary(model_data))

# Step 5: Modeling

# 5-fold cv setup
set.seed(123)
train_control <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE
)

features <- setdiff(names(model_data), "ObesityLevel")
target <- "ObesityLevel"

# Decision tree
dt_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))

model_dt <- train(
  as.formula(paste(target, "~ .")),
  data = model_data,
  method = "rpart",
  trControl = train_control,
  tuneGrid = dt_grid,
  control = rpart.control(
    maxdepth = 5,
    minsplit = 20
  )
)

# Random Forest
set.seed(42)
model_rf <- train(as.formula(paste(target, "~ .")),
                  data = model_data,
                  method = "rf",
                  trControl = train_control,
                  ntree = 100)

# Multinomial Logistic Regression
model_mlr <- train(as.formula(paste(target, "~ .")),
                   data = model_data,
                   method = "multinom",
                   trControl = train_control,
                   trace = FALSE)

# Confusion Matrix, ROC/AUC, F1 scores
conf_auc <- function(model, model_name) {
  cat(paste0("\n Confusion Matrix: \n", model_name))
  cm <- confusionMatrix(model$pred$pred, model$pred$obs)
  print(cm)
  
  pred_df <- model$pred
  pred_df <- pred_df[order(pred_df$rowIndex), ]
  true_labels <- pred_df$obs
  prob_pred <- pred_df[, levels(true_labels)]
  true_mat <- model.matrix(~ obs - 1, data = pred_df)
  colnames(true_mat) <- gsub("^obs", "", colnames(true_mat))
  
  # Store ROC data
  roc_list <- list()
  auc_df <- data.frame()
  
  for (class in colnames(true_mat)) {
    roc_obj <- tryCatch({
      roc(response = true_mat[, class], predictor = prob_pred[[class]])
    }, error = function(e) return(NULL))
    
    if (!is.null(roc_obj)) {
      auc_val <- auc(roc_obj)
      auc_df <- rbind(auc_df, data.frame(Class = class, AUC = round(auc_val, 4)))
      roc_data <- data.frame(
        FPR = 1 - roc_obj$specificities,
        TPR = roc_obj$sensitivities,
        Class = class
      )
      roc_list[[class]] <- roc_data
    }
  }
  
  # Combine all ROC curves into one data frame
  roc_df <- do.call(rbind, roc_list)
  
  # Plot combined ROC curves using ggplot2
  p <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Class)) +
    geom_line(size = 1) +
    geom_abline(linetype = "dashed", color = "gray") +
    labs(title = paste("ROC Curves -", model_name),
         x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal() +
    coord_equal()
  print(p)
  
  # Print AUCs
  cat(paste0("\nPer-Class AUC: \n", model_name))
  print(auc_df)
  
  # F1 Score (macro-averaged)
  f1_values <- sapply(levels(true_labels), function(class) {
    F1_Score(y_true = as.factor(true_labels == class),
             y_pred = as.factor(pred_df$pred == class),
             positive = "TRUE")
  })
  f1_macro <- mean(f1_values)
  cat(paste0("\nMacro-Averaged F1 Score for ", model_name, ": ", round(f1_macro, 4), "\n"))
  
}

conf_auc(model_dt, "Decision Tree")
conf_auc(model_rf, "Random Forest")
conf_auc(model_mlr, "Multinomial Logistic Regression")

# Step 6: Interpretability & Fairness
# LIME
# Preparing feature matrix
X_lime <- model_data[, features]

# Creating LIME explainer
explainer_rf <- lime(X_lime, model_rf, bin_continuous = TRUE, n_bins = 4)

obs_index <- 1 #1st observation
observation <- X_lime[obs_index, , drop = FALSE]

all_labels <- levels(model_data$ObesityLevel)

# Generating LIME explanation for all labels
explanation <- lime::explain(
  observation,
  explainer = explainer_rf,
  n_features = 5,
  labels = all_labels
)

plot_features(explanation)

# SHAP
# Define class labels
class_labels <- levels(model_data$ObesityLevel)

# Wrapper creator for class-specific prediction
make_wrapper <- function(class) {
  function(model, newdata) predict(model, newdata, type = "prob")[, class]
}

# Function to compute SHAP values and return long-format dataframe
compute_shap_long <- function(class) {
  shap <- explain(model_rf, X = X_lime, pred_wrapper = make_wrapper(class), nsim = 50)
  as.data.frame(shap) %>%
    mutate(obs = row_number()) %>%
    pivot_longer(-obs, names_to = "feature", values_to = "value") %>%
    mutate(class = class)
}

# Compute SHAP values for all classes
shap_all_df <- bind_rows(lapply(class_labels, compute_shap_long))

# Global mean absolute SHAP values
shap_summary <- shap_all_df %>%
  group_by(class, feature) %>%
  summarise(mean_abs_shap = mean(abs(value)), .groups = "drop")

ggplot(shap_summary, aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap, fill = class)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Global SHAP Values per Class", x = "Feature", y = "Mean(|SHAP Value|)") +
  theme_minimal()