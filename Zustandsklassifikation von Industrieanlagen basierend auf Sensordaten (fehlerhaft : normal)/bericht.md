# Experimentbericht

## Zustandsklassifikation von Industrieanlagen basierend auf Sensordaten (fehlerhaft : normal)

---

## 1. Experimentübersicht

### 1.1 Forschungsziel

Ziel dieses Forschung ist die Entwicklung eines maschinellen Lernmodells zur Echtzeit-Klassifikation des Betriebszustands von Industrieanlagen basierend auf Sensordaten, zur Unterscheidung zwischen Normalzustand (Normal) und Fehlerzustand (Fehler).

### 1.2 Datenquelle

Der Datensatz wurde in der Programmiersprache R simuliert und enthält folgende Merkmale:

- **Vibration (vibration)**
- **Akustiksignal (acoustic)**
- **Temperatur (temperature)**
- **Strom (current)**
- **Intrinsische Modenfunktions-Zerlegungsmerkmale (IMF_1, IMF_2, IMF_3)**

### 1.3 Anlageninformationen

Der Datensatz umfasst 4 Industrieanlagen mit den Nummern M01, M02, M03 und M04.

---

## 2. Datensatzbeschreibung

### 2.1 Datenumfang

| Metrik | Wert |
|-------|------|
| Gesamtanzahl Proben | 2000 |
| Trainingsmenge | 1600 (80%) |
| Testmenge | 400 (20%) |
| Zeitraum | 2024-07-01 08:00 bis 2024-07-02 17:19 |

### 2.2 Fehlerverteilung

| Zustand | Probenanzahl | Anteil |
|---------|-------------|-------|
| Normal (0) | 1516 | 75,8% |
| Fehler (1) | 484 | 24,2% |

**Gesamtfehlerrate: 24,2%**

### 2.3 Fehlerstatistik je Anlage

| Anlage | Normalproben | Fehlerproben | Fehlerrate |
|--------|-------------|--------------|------------|
| M01 | 377 | 123 | 24,6% |
| M02 | 378 | 122 | 24,4% |
| M03 | 380 | 120 | 24,0% |
| M04 | 381 | 119 | 23,8% |

Die Fehlerverteilung je Anlage ist weitgehend ausgeglichen, mit Fehlerraten zwischen 23,8% und 24,6%.

---

## 3. Explorative Datenanalyse (EDA)

### 3.1 Merkmalsstatistik

Die Verteilung der Sensormerkmale unterscheidet sich signifikant zwischen Normalzustand und Fehlerzustand. Das Temperatur-Boxplot zeigt, dass die Temperaturwerte im Fehlerzustand deutlich höher sind als im Normalzustand.

### 3.2 Korrelationsanalyse

Die Korrelationsmatrix der Merkmale zeigt:
- vibration und acoustic sind positiv korreliert
- temperature und current sind positiv korreliert
- Die IMF-Merkmale untereinander zeigen gewisse Korrelation

(siehe `eda_korrelations_heatmap.png`)

---

## 4. Modellaufbau und Training

### 4.1 Datenvorverarbeitung

- Zeimerkmalsextraktion: Stunde (hour), Wochentag (weekday), Wochenende (is_weekend)
- Anlagen-ID zu numerischer Kodierung konvertiert
- Trainings- und Testmenge zeitlich geordnet aufgeteilt

### 4.2 Modell 1: Logistische Regression (Logistic Regression)

Als Basismodell wird die logistische Regression mit allen Merkmalen für die binäre Klassifikation verwendet.

### 4.3 Modell 2: XGBoost (kostensensitiv)

Unter Anwendung einer kostensensitiven Lernstrategie wird die Fehlerklasse gewichtet mit `3 × (Negativproben/Positivproben)`, um die Recall-Rate der Minderheitsklasse (Fehler) zu erhöhen.

**XGBoost-Hyperparameter:**
- max_depth: 4
- eta: 0,1
- subsample: 0,8
- colsample_bytree: 0,8
- early_stopping_rounds: 20

---

## 5. Experimentergebnisse

### 5.1 Modellleistungsvergleich

| Modell | Genauigkeit (Accuracy) | Präzision (Precision) | Recall (Recall) | F1-Score |
|--------|----------------------|---------------------|-----------------|----------|
| Logistische Regression | 1,000 | 1,000 | 1,000 | 1,000 |
| **XGBoost** | **0,988** | **0,984** | **0,976** | **0,980** |

### 5.2 XGBoost-Konfusionsmatrix (Testmenge n=400)

|  | Vorhersage: Normal | Vorhersage: Fehler |
|--|--------------------|--------------------|
| **Tatsächlich: Normal** | 273 (TN) | 2 (FP) |
| **Tatsächlich: Fehler** | 3 (FN) | 122 (TP) |

- Wahr Negativ (TN): 273
- Falsch Positiv (FP): 2
- Falsch Negativ (FN): 3
- Wahr Positiv (TP): 122

### 5.3 Merkmalsbedeutung (XGBoost)

Die nach Gain-Werten sortierte Merkmalsbedeutung zeigt:
1. vibration
2. temperature
3. current
4. acoustic
5. IMF_1
6. IMF_2
7. IMF_3

(siehe `dashboard_feature_importance.png`)

### 5.4 ROC-Kurve

Die ROC-Kurve des XGBoost-Modells erreicht eine Fläche unter der Kurve (AUC) von 0,999, was eine außergewöhnliche Klassifikationsleistung zeigt.

(siehe `dashboard_roc_curve.png`)

---

## 6. Schlussfolgerung

### 6.1 Wesentliche Erkenntnisse

1. **Hohe Fehlererkennungsrate**: Im simulierten Datensatz machen Fehlerproben 24,2% der Gesamtmenge aus. Das XGBoost-Modell erreicht eine Recall-Rate von 97,6% und kann den Großteil der Fehlerzustände effektiv identifizieren.

2. **Niedrige Fehlalarmrate**: In der Testmenge treten nur 2 falsch positive Fälle auf (Normalzustand wird als Fehler vorhergesagt), was einer Fehlalarmrate von nur 0,5% entspricht.

3. **Merkmalsbedeutung**: Vibration (vibration), Temperatur (temperature) und Strom (current) sind die drei wichtigsten Merkmale zur Unterscheidung des Anlagenzustands.

4. **Modellvergleich**: Die logistische Regression zeigt auf diesem Datensatz eine perfekte Leistung (mögliches Overfitting-Risiko). XGBoost bietet einen praktischeren Wert, wobei die kostensensitive Strategie die Recall-Rate der Fehlererkennung effektiv verbessert.

### 6.2 Empfehlungen für die praktische Anwendung

- XGBoost-Modell für die Echtzeit-Zustandsüberwachung verwenden
- Besonderes Augenmerk auf anomalische Änderungen bei Temperatur- und Vibrationssensoren legen
- Der Klassifikationsschwellenwert kann je nach tatsächlicher Situation angepasst werden, um Präzision und Recall auszubalancieren

---

## 7. Erläuterung der Ausgabedateien

| Dateiname | Beschreibung |
|-----------|--------------|
| `simulated_raw_data.csv` | Ursprünglicher simulierter Datensatz (2000 Zeilen) |
| `predictions_for_dashboard.csv` | Vorhersageergebnisse der Testmenge (400 Zeilen) |
| `model_comparison.csv` | Modellleistungsvergleichsmetriken |
| `eda_temperatur_boxplot.png` | Temperatur-Boxplot |
| `eda_korrelations_heatmap.png` | Merkmalskorrelations-Heatmap |
| `eda_fehlerrate_pro_maschine.png` | Balkendiagramm der Fehlerrate je Anlage |
| `dashboard_feature_importance.png` | Merkmalsbedeutungsdiagramm |
| `dashboard_confusion_matrix.png` | Konfusionsmatrix-Heatmap |
| `dashboard_roc_curve.png` | ROC-Kurve |
| `dashboard_time_series_prob.png` | Vorhersagewahrscheinlichkeits-Zeitreihendiagramm |

---

**Berichtsdatum:** 2024-07-02

**Experimentumgebung:** R (tidyverse, xgboost, caret, pROC)