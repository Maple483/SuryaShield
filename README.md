# SuryaShield
Geoeffective ICMEs prediction

# AI-Based ICME Geoeffectiveness Prediction

An AI-driven early-warning system for predicting whether an incoming **Interplanetary Coronal Mass Ejection (ICME)** will produce a significant geomagnetic storm using only the **first 3 hours of in-situ solar-wind observations** after shock arrival.

## Overview

Geomagnetic storms caused by ICMEs can disrupt **satellite operations, communication systems, navigation, power infrastructure, and other space-based technologies**.
Current forecasting approaches often rely on CME observations and physics-based propagation models to estimate arrival time and characteristics. However, determining the **eventual geoeffectiveness and severity** of an ICME remains challenging.

Our approach uses machine learning to identify early solar-wind signatures associated with geoeffective ICMEs.

### Core Idea

```text
ICME Shock Arrival
        ↓
First 3 Hours of Solar-Wind Data
        ↓
Data Preprocessing & Feature Engineering
        ↓
LSTM Time-Series Model
        ↓
Geoeffectiveness Prediction
        ↓
Early Warning
```

The model predicts whether the ICME is likely to produce a significant geomagnetic storm, defined in this project as:

**Minimum Dst ≤ −50 nT**

---

The LSTM model captures temporal patterns in parameters such as:

* Magnetic-field magnitude (`B`)
* `Bx`, `By`, `Bz`
* Solar-wind speed
* Proton density
* Proton temperature
* Dynamic pressure
* Convective electric field (`Ey`)
* Plasma beta
* Alfvén Mach number

This transforms early satellite observations into an actionable prediction of potential geomagnetic impact.

---

## Machine Learning

The project uses an **LSTM (Long Short-Term Memory)** neural network because solar-wind measurements form a time series, where the evolution of parameters over several observations can be more informative than individual measurements.

### Input

The first **3 hours of in-situ solar-wind observations** following ICME shock arrival.

### Output

Binary classification:

```text
0 → Non-geoeffective ICME
1 → Geoeffective ICME
```

An ICME is considered geoeffective when:

```text
Minimum Dst ≤ −50 nT
```

---

## Dataset

### Richardson & Cane ICME Catalog

Used to identify ICME events, including their shock arrival and event boundaries, and each event's minimum Dst index.

### NASA OMNI Dataset

Provides solar-wind and interplanetary magnetic-field measurements near Earth.



### Historical Coverage

The current dataset covers approximately:

**1996–2025**

This provides multiple solar cycles and a broad range of ICME conditions for model development and evaluation.

---

## Model Performance

Current LSTM performance after threshold tuning:

| Metric    |    Result |
| --------- | --------: |
| Accuracy  | **73.0%** |
| Precision | **69.7%** |
| Recall    | **76.7%** |
| F1 Score  | **73.0%** |
| ROC-AUC   | **0.804** |

The **76.7% recall** is particularly important for an early-warning application because the system prioritizes detecting potentially geoeffective events.

---

## System Architecture

```text
Richardson & Cane Catalog
          │
          ▼
    Identify ICME Events
          │
          ▼
      NASA OMNI Data
          │
          ▼
 Extract First 3 Hours
          │
          ▼
 Preprocessing & Features
          │
          ▼
       LSTM Model
          │
          ▼
 Geoeffectiveness Prediction
          │
          ▼
      Early Warning
```

## Running the Project

Run the main application:

```bash
python app.py
```

The application provides an interface for interacting with the trained prediction system.


