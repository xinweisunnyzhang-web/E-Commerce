# =====================================================================
# Auftrag für OpenClaw: Simulierte Daten generieren und vollständige Analyse durchführen
# Zustandsklassifikation von Industrieanlagen (fehlerhaft/normal)
# Ausgabe: Grafiken (PNG) und CSV-Dateien im Arbeitsverzeichnis
# =====================================================================

# ----------------------------
# 0. Pakete prüfen und installieren (falls nötig)
# ----------------------------
required_packages <- c("tidyverse", "lubridate", "caret", "xgboost", 
                       "ggplot2", "corrplot", "pROC", "patchwork")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Zufallsseed für reproduzierbare Ergebnisse
set.seed(2024)

# ----------------------------
# 1. Simulierte Sensordaten erzeugen
# ----------------------------
generate_data <- function(n = 2000, n_machines = 4) {
  timestamps <- seq(as.POSIXct("2024-07-01 08:00:00"), by = "1 min", length.out = n)
  machine_id <- rep(paste0("M", sprintf("%02d", 1:n_machines)), length.out = n)
  
  # Normal state parameters
  normal_mean <- c(vibration = 0.8, acoustic = 0.6, temperature = 65, 
                   current = 12, IMF_1 = 0.2, IMF_2 = 0.1, IMF_3 = 0.05)
  normal_sd   <- c(0.2, 0.15, 5, 2, 0.05, 0.03, 0.02)
  
  # Fault shift
  fault_shift <- c(vibration = 0.4, acoustic = 0.3, temperature = 12,
                   current = 4, IMF_1 = 0.15, IMF_2 = 0.1, IMF_3 = 0.08)
  
  # Generate labels (clustered faults, about 20%)
  label <- integer(n)
  fault_segments <- sample(1:(n - 20), size = floor(n * 0.2 / 20), replace = FALSE)
  for (seg in fault_segments) {
    len <- sample(10:40, 1)
    end <- min(seg + len, n)
    label[seg:end] <- 1
  }
  if (mean(label) < 0.15) {
    extra <- sample(which(label == 0), size = round(n * 0.05))
    label[extra] <- 1
  }
  
  # Generate sensor values
  sensor_data <- matrix(NA, nrow = n, ncol = 7)
  for (i in 1:n) {
    if (label[i] == 0) {
      sensor_data[i, ] <- rnorm(7, normal_mean, normal_sd)
    } else {
      sensor_data[i, ] <- rnorm(7, normal_mean + fault_shift, normal_sd * 1.3)
    }
  }
  sensor_data <- abs(sensor_data)
  colnames(sensor_data) <- names(normal_mean)
  
  df <- data.frame(timestamp = timestamps,
                   machine_id = machine_id,
                   sensor_data,
                   label = label)
  df <- df %>% arrange(timestamp)
  return(df)
}

cat("Generiere simulierte Daten...\n")
data_raw <- generate_data(n = 2000, n_machines = 4)
cat("Dimensionen:", dim(data_raw), "\n")
cat("Fehleranteil:", mean(data_raw$label), "\n")
write.csv(data_raw, "simulated_raw_data.csv", row.names = FALSE)

# ----------------------------
# 2. Feature Engineering
# ----------------------------
data_processed <- data_raw %>%
  mutate(timestamp = ymd_hms(timestamp),
         machine_id = as.factor(machine_id),
         label = as.factor(label)) %>%
  arrange(timestamp) %>%
  mutate(hour = hour(timestamp),
         weekday = wday(timestamp, week_start = 1),
         is_weekend = ifelse(weekday %in% c(6,7), 1, 0)) %>%
  select(-timestamp)

# Train/test split (temporal order)
n_total <- nrow(data_processed)
train_idx <- 1:floor(0.8 * n_total)
train <- data_processed[train_idx, ]
test  <- data_processed[-train_idx, ]

X_train <- train %>% select(-label) %>% mutate(across(where(is.factor), as.numeric))
y_train <- as.numeric(as.character(train$label))
X_test  <- test  %>% select(-label) %>% mutate(across(where(is.factor), as.numeric))
y_test  <- as.numeric(as.character(test$label))

X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

# ----------------------------
# 3. Explorative Datenanalyse (EDA) – Grafiken speichern
# ----------------------------
# 3.1 Boxplot Temperatur
p1 <- ggplot(data_processed, aes(x = label, y = temperature, fill = label)) +
  geom_boxplot() + 
  labs(title = "Temperatur nach Maschinenzustand", x = "Label (0=Normal, 1=Fehler)", y = "Temperatur (°C)") +
  theme_minimal()
ggsave("eda_temperatur_boxplot.png", p1, width = 6, height = 4)

# 3.2 Korrelationsmatrix
num_features <- data_processed %>% select(vibration, acoustic, temperature, current, IMF_1, IMF_2, IMF_3)
cor_matrix <- cor(num_features)
png("eda_korrelations_heatmap.png", width = 800, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, title = "Merkmal-Korrelationen")
dev.off()

# 3.3 Fehlerrate pro Maschine
fault_rate <- data_processed %>%
  group_by(machine_id, label) %>%
  summarise(count = n(), .groups = "drop") %>%
  pivot_wider(names_from = label, values_from = count, values_fill = 0) %>%
  mutate(fault_rate = `1` / (`0` + `1`))
p2 <- ggplot(fault_rate, aes(x = machine_id, y = fault_rate, fill = machine_id)) +
  geom_col() + labs(title = "Fehlerrate pro Maschine") + theme_minimal()
ggsave("eda_fehlerrate_pro_maschine.png", p2, width = 6, height = 4)

# ----------------------------
# 4. Logistische Regression (Baseline)
# ----------------------------
logit_df <- cbind(y_train, X_train)
colnames(logit_df)[1] <- "label"
logit_model <- glm(label ~ ., data = logit_df, family = binomial)

prob_logit <- predict(logit_model, newdata = X_test, type = "response")
pred_logit <- ifelse(prob_logit > 0.5, 1, 0)
cm_logit <- caret::confusionMatrix(as.factor(pred_logit), as.factor(y_test), positive = "1")

# ----------------------------
# 5. XGBoost (cost-sensitive)
# ----------------------------
neg <- sum(y_train == 0)
pos <- sum(y_train == 1)
scale_pos <- (neg / pos) * 3   # cost weight for faults

params <- list(objective = "binary:logistic",
               eval_metric = "logloss",
               max_depth = 4, eta = 0.1,
               scale_pos_weight = scale_pos,
               subsample = 0.8, colsample_bytree = 0.8)

xgb_model <- xgboost(data = X_train_mat, label = y_train,
                     params = params, nrounds = 200,
                     early_stopping_rounds = 20, verbose = 0)

importance <- xgb.importance(model = xgb_model, feature_names = colnames(X_train_mat))

prob_xgb <- predict(xgb_model, X_test_mat)
pred_xgb <- ifelse(prob_xgb > 0.5, 1, 0)
cm_xgb <- caret::confusionMatrix(as.factor(pred_xgb), as.factor(y_test), positive = "1")

# ----------------------------
# 6. Dashboard-Grafiken (alle mit R)
# ----------------------------
# 6.1 Feature Importance
imp_df <- importance %>% 
  mutate(Feature = reorder(Feature, Gain)) %>%
  head(10)
p_imp <- ggplot(imp_df, aes(x = Gain, y = Feature)) +
  geom_col(fill = "steelblue") +
  labs(title = "XGBoost - Feature Importance", x = "Gain") +
  theme_minimal()
ggsave("dashboard_feature_importance.png", p_imp, width = 7, height = 5)

# 6.2 Time series of predicted probabilities
test_with_time <- test %>%
  mutate(timestamp = data_raw$timestamp[(train_idx[length(train_idx)]+1):n_total],
         pred_prob = prob_xgb,
         true_label = y_test)
p_ts <- ggplot(test_with_time, aes(x = timestamp, y = pred_prob, color = machine_id)) +
  geom_line(alpha = 0.7) + 
  labs(title = "Vorhersage der Fehlerwahrscheinlichkeit (Testsatz)", y = "Wahrscheinlichkeit") +
  theme_minimal() +
  facet_wrap(~machine_id, ncol = 2)
ggsave("dashboard_time_series_prob.png", p_ts, width = 10, height = 6)

# 6.3 Confusion Matrix heatmap
cm_mat <- cm_xgb$table
cm_df <- as.data.frame(cm_mat)
colnames(cm_df) <- c("Prediction", "Reference", "Freq")
p_cm <- ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix - XGBoost") +
  theme_minimal()
ggsave("dashboard_confusion_matrix.png", p_cm, width = 5, height = 4)

# 6.4 ROC curve
roc_obj <- pROC::roc(y_test, prob_xgb)
roc_df <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)
p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", size = 1) +
  geom_abline(linetype = "dashed", alpha = 0.5) +
  labs(title = paste0("ROC-Kurve (AUC = ", round(auc(roc_obj), 3), ")"),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()
ggsave("dashboard_roc_curve.png", p_roc, width = 5, height = 5)

# ----------------------------
# 7. Modellvergleich und Ausgabe
# ----------------------------
extract_metrics <- function(cm) {
  data.frame(
    Accuracy = round(cm$overall["Accuracy"], 3),
    Precision = round(cm$byClass["Precision"], 3),
    Recall = round(cm$byClass["Sensitivity"], 3),
    F1 = round(cm$byClass["F1"], 3)
  )
}
comparison <- rbind(extract_metrics(cm_logit), extract_metrics(cm_xgb))
rownames(comparison) <- c("Logistic Regression", "XGBoost (cost-sensitive)")
write.csv(comparison, "model_comparison.csv", row.names = TRUE)

# Save predictions for later
pred_output <- test %>%
  mutate(timestamp = data_raw$timestamp[(train_idx[length(train_idx)]+1):n_total],
         true_label = y_test,
         xgb_prob = prob_xgb,
         xgb_pred = pred_xgb) %>%
  select(timestamp, machine_id, vibration, acoustic, temperature, current,
         IMF_1, IMF_2, IMF_3, true_label, xgb_prob, xgb_pred)
write.csv(pred_output, "predictions_for_dashboard.csv", row.names = FALSE)

# Console output
cat("\n========== Logistic Regression ==========\n")
print(cm_logit)
cat("\n========== XGBoost (cost-sensitive) ==========\n")
print(cm_xgb)
cat("\n========== Model Comparison ==========\n")
print(comparison)
cat("\nAlle Grafiken und Ausgabedateien wurden erstellt. Projekt abgeschlossen.\n")