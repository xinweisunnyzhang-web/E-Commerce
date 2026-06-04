# ============================================================
# Vorhersage von Wiederkäufen im E-Commerce: LR vs. XGBoost
# Geeignet für Veröffentlichungen: Hyperparameter + AME + SHAP + 10:1 Kostenanalyse
# 
# ============================================================

library(dplyr)
library(caret)
library(pROC)
library(readxl)
library(ggplot2)
library(xgboost)                             
library(margins)  # Marginale Effekte
library(shapviz)  # SHAP-Visualisierung

# ====================== 1. Daten einlesen und Features konstruieren ======================
path <- file.path("/Users/jingzhang/Desktop/E-Commerce-PlattformenData",
                  "2026-4-28 00.41 Kosten schwellwert E-commerce",
                  "user_log_format1.xlsx")
df <- read_xlsx(path)

df$time_stamp <- as.numeric(df$time_stamp)

# Merkmale pro Nutzer-Händler-Paar erstellen
user_feature <- df %>%
  group_by(user_id, seller_id) %>%
  summarise(
    click = sum(action_type == 0),
    cart  = sum(action_type == 1),
    fav   = sum(action_type == 2),
    days  = max(time_stamp) - min(time_stamp) + 1,
    label = ifelse(sum(action_type == 3) >= 2, 1, 0),
    .groups = "drop"
  )

# IDs entfernen, nur Features und Label behalten
model_data <- user_feature %>% select(-user_id, -seller_id)

# Trainings-/Testsplit
set.seed(123)
trainIndex <- createDataPartition(model_data$label, p = 0.7, list = FALSE)
train <- model_data[trainIndex, ]
test  <- model_data[-trainIndex, ]

# Label als Faktor kodieren
train$label <- factor(train$label, levels = c(0,1))
test$label  <- factor(test$label, levels = c(0,1))

true_test <- as.numeric(test$label) - 1



# ====================== 2. Logistische Regression (mit definierten Hyperparametern) ======================
pos_weight <- (1 - mean(train$label == 1)) / mean(train$label == 1)

lr <- glm(
  label ~ click + cart + fav + days, 
  data = train, 
  family = binomial(link = "logit"),  # Hyperparameter: Logit-Link
  control = glm.control(maxit = 100), # Hyperparameter: maximale Iterationen
  weights = ifelse(train$label == 1, pos_weight, 1)
)

pred_prob_lr <- predict(lr, test, type = "response")
roc_lr <- roc(true_test, pred_prob_lr)
auc_lr <- auc(roc_lr)



# ====================== 3. XGBoost  ======================
X_train <- train %>% select(click, cart, fav, days) %>% as.matrix()
y_train <- as.numeric(train$label) - 1
X_test  <- test %>% select(click, cart, fav, days) %>% as.matrix()
y_test  <- true_test

scale_pos_weight <- sum(y_train==0)/sum(y_train==1)

dtrain <- xgb.DMatrix(X_train, label=y_train)
dtest  <- xgb.DMatrix(X_test, label=y_test)

# ====================== feste Hyperparameter======================
params <- list(
  objective           = "binary:logistic",
  eval_metric         = "auc",
  max_depth           = 3,      # festgelegt
  eta                 = 0.1,    # festgelegt
  subsample           = 0.8,    # festgelegt
  colsample_bytree    = 0.8,    # festgelegt
  scale_pos_weight    = scale_pos_weight
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(test=dtest),
  verbose = 0
)

pred_prob_xgb <- predict(xgb_model, dtest)
roc_xgb <- roc(y_test, pred_prob_xgb)
auc_xgb <- auc(roc_xgb)



# ====================== 4. Ökonomische Analyse 1: Marginale Effekte (AME) der LR ======================
cat("\n===== Marginale Effekte (Average Marginal Effects) =====\n")
ame <- margins(lr)
summary(ame)



# ====================== 5. Ökonomische Analyse 2: SHAP-Interpretation (XGBoost) ======================
shap <- shapviz(xgb_model, X_pred = X_test)
sv_importance(shap, kind = "both")     # Feature Importance Plot



# ====================== 6. Ökonomische Analyse 3: 10:1 Kostenanalyse (Kernstück) ======================
cost_FN <- 10   # Verpasster Wiederkäufer: Verlust 10 €
cost_FP <- 1    # Fälschlich ausgesandter Gutschein: Verlust 1 €

calc_cost <- function(prob, threshold, true_y) {
  pred <- ifelse(prob >= threshold, 1, 0)
  FN <- sum(pred == 0 & true_y == 1)
  FP <- sum(pred == 1 & true_y == 0)
  return(FN * cost_FN + FP * cost_FP)
}

thresholds <- seq(0.1, 0.9, 0.01)
cost_lr  <- sapply(thresholds, function(t) calc_cost(pred_prob_lr, t, true_test))
cost_xgb <- sapply(thresholds, function(t) calc_cost(pred_prob_xgb, t, true_test))

best_t_lr  <- thresholds[which.min(cost_lr)]
best_t_xgb <- thresholds[which.min(cost_xgb)]
min_cost_lr  <- min(cost_lr)
min_cost_xgb <- min(cost_xgb)

cat("\n===== 10:1 Kostenanalyse Ergebnisse =====\n")
cat("LR Minimale Kosten:", min_cost_lr, " | Optimaler Schwellenwert:", best_t_lr, "\n")
cat("XGBoost Minimale Kosten:", min_cost_xgb, " | Optimaler Schwellenwert:", best_t_xgb, "\n")

# Kostenkurve
plot(thresholds, cost_lr, type="l", col="blue", lwd=2,
     main="Kosten vs. Schwellwert (FN:10€, FP:1€)",
     xlab="Schwellwert", ylab="Gesamtkosten")
lines(thresholds, cost_xgb, col="red", lwd=2)
legend("topright", c("LR","XGBoost"), col=c("blue","red"), lwd=2)



# ====================== 7. Modellgütemaße ausgeben ======================
# LR Metriken (Youden-optimaler Schwellwert)
best_th_lr <- coords(roc_lr, "best")$threshold
pred_class_lr <- factor(ifelse(pred_prob_lr > best_th_lr,1,0), levels=c(0,1))
cm_lr <- table(test$label, pred_class_lr)
TN_lr <- cm_lr[1,1]; FP_lr <- cm_lr[1,2]; FN_lr <- cm_lr[2,1]; TP_lr <- cm_lr[2,2]
accuracy_lr <- (TP_lr+TN_lr)/(TP_lr+TN_lr+FP_lr+FN_lr)
precision_lr <- TP_lr/(TP_lr+FP_lr)
recall_lr <- TP_lr/(TP_lr+FN_lr)
f1_lr <- 2*precision_lr*recall_lr/(precision_lr+recall_lr)

# XGBoost Metriken
best_th_xgb <- coords(roc_xgb, "best")$threshold
pred_class_xgb <- ifelse(pred_prob_xgb > best_th_xgb,1,0)
cm_xgb <- table(y_test, pred_class_xgb)
TN_xgb <- cm_xgb[1,1]; FP_xgb <- cm_xgb[1,2]; FN_xgb <- cm_xgb[2,1]; TP_xgb <- cm_xgb[2,2]
accuracy_xgb <- (TP_xgb+TN_xgb)/(TP_xgb+TN_xgb+FP_xgb+FN_xgb)
precision_xgb <- TP_xgb/(TP_xgb+FP_xgb)
recall_xgb <- TP_xgb/(TP_xgb+FN_xgb)
f1_xgb <- 2*precision_xgb*recall_xgb/(precision_xgb+recall_xgb)

cat("\n===== Logistische Regression =====\n")
cat("AUC =", round(auc_lr,3), "\n")
print(cm_lr)

cat("\n===== XGBoost =====\n")
cat("AUC =", round(auc_xgb,3), "\n")
print(cm_xgb)



# ====================== 8. ROC-Kurve ======================
roc_list <- list(Logistic = roc_lr, XGBoost = roc_xgb)
ggroc(roc_list, legacy.axes=TRUE, size=1.2) +
  geom_abline(linetype="dashed", color="gray") +
  scale_color_manual(values=c("steelblue","firebrick")) +
  labs(title="ROC Kurvenvergleich", x="FPR", y="TPR") +
  theme_minimal() +
  annotate("text", x=0.75,y=0.25, label=paste("LR AUC=",round(auc_lr,3)), col="steelblue") +
  annotate("text", x=0.75,y=0.20, label=paste("XGB AUC=",round(auc_xgb,3)), col="firebrick")


# ====================== 9. Erweiterte Kostenanalyse: Verschiedene Kostenverhältnisse ======================
# Beantwortet die Frage: Ab wann lohnt sich welches Modell wirklich?

FN_werte <- c(5, 10, 15, 20, 30, 50)
FP_fix <- 1

ergebnisse <- data.frame()

for (FN_aktuell in FN_werte) {
  
  calc_cost_aktuell <- function(prob, threshold, true_y) {
    pred <- ifelse(prob >= threshold, 1, 0)
    FN <- sum(pred == 0 & true_y == 1)
    FP <- sum(pred == 1 & true_y == 0)
    return(FN * FN_aktuell + FP * FP_fix)
  }
  
  cost_lr_aktuell <- sapply(thresholds, function(t) calc_cost_aktuell(pred_prob_lr, t, true_test))
  cost_xgb_aktuell <- sapply(thresholds, function(t) calc_cost_aktuell(pred_prob_xgb, t, true_test))
  
  min_cost_lr <- min(cost_lr_aktuell)
  min_cost_xgb <- min(cost_xgb_aktuell)
  best_t_lr <- thresholds[which.min(cost_lr_aktuell)]
  best_t_xgb <- thresholds[which.min(cost_xgb_aktuell)]
  
  besser <- ifelse(min_cost_xgb < min_cost_lr, "XGBoost", ifelse(min_cost_lr < min_cost_xgb, "LR", "gleich"))
  
  ergebnisse <- rbind(ergebnisse, data.frame(
    FN_Kosten = FN_aktuell,
    FP_Kosten = FP_fix,
    Kostenverhaeltnis = FN_aktuell / FP_fix,
    LR_min_Kosten = round(min_cost_lr, 0),
    XGB_min_Kosten = round(min_cost_xgb, 0),
    besser = besser,
    LR_opt_Schwelle = round(best_t_lr, 2),
    XGB_opt_Schwelle = round(best_t_xgb, 2)
  ))
}

cat("\n===== Kostenverhältnis-Vergleich (FN variabel, FP=1) =====\n")
print(ergebnisse)

# ====================== 10. Diagramm: Kostenverhältnis vs. minimale Gesamtkosten ======================
plot(ergebnisse$Kostenverhaeltnis, ergebnisse$LR_min_Kosten, 
     type = "b", col = "blue", pch = 16, lwd = 2,
     xlab = "Kostenverhältnis (FN : FP)", ylab = "Minimale Gesamtkosten (€)",
     main = "Vergleich der minimalen Gesamtkosten bei verschiedenen Kostenverhältnissen",
     ylim = range(c(ergebnisse$LR_min_Kosten, ergebnisse$XGB_min_Kosten)))
lines(ergebnisse$Kostenverhaeltnis, ergebnisse$XGB_min_Kosten, 
      type = "b", col = "red", pch = 17, lwd = 2)
legend("topright", 
       legend = c("Logistische Regression", "XGBoost"),
       col = c("blue", "red"), 
       lwd = 2, 
       pch = c(16, 17),      
       lty = 1,             
       seg.len = 3)          
grid()

# Kritischen Punkt finden, ab dem XGBoost wirtschaftlicher wird
kritisch <- ergebnisse[which(ergebnisse$besser == "XGBoost")[1], ]
if (!is.na(kritisch$Kostenverhaeltnis)) {
  cat("\n===== Kritischer Punkt =====\n")
  cat("Ab einem Kostenverhältnis von", kritisch$Kostenverhaeltnis, ":1 wird XGBoost wirtschaftlicher.\n")
} else {
  cat("\nKein kritischer Punkt gefunden – LR bleibt bei allen getesteten Verhältnissen besser.\n")
}