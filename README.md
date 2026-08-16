# SuryaShield
Geoeffective ICMEs prediction

# AI-Based ICME Geoeffectiveness Prediction

An AI-driven early-warning system for predicting whether an incoming **Interplanetary Coronal Mass Ejection (ICME)** will produce a significant geomagnetic storm using only the **first 3 hours of in-situ solar-wind observations** after shock arrival.

## 🚀 Overview

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

## 💡 Key Innovation

The most novel aspect of the project is **early geoeffectiveness prediction**.

Instead of waiting for an ICME to complete its interaction with Earth's magnetosphere, the system uses only the **first three hours of observed solar-wind conditions** to predict its eventual geomagnetic impact.

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

## 🧠 Machine Learning Approach

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

## 📊 Dataset

The project integrates three major sources of space-weather data:

### Richardson & Cane ICME Catalog

Used to identify ICME events, including their shock arrival and event boundaries.

### NASA OMNI Dataset

Provides solar-wind and interplanetary magnetic-field measurements near Earth.

### Kyoto Dst Index

Used to determine the geomagnetic response associated with each ICME and generate the geoeffectiveness labels.

### Historical Coverage

The current dataset covers approximately:

**1996–2025**

This provides multiple solar cycles and a broad range of ICME conditions for model development and evaluation.

---

## 📈 Model Performance

Current LSTM performance after threshold tuning:

| Metric    |    Result |
| --------- | --------: |
| Accuracy  | **69.8%** |
| Precision | **64.1%** |
| Recall    | **87.2%** |
| F1 Score  | **73.9%** |
| ROC-AUC   | **78.4%** |

The **87.2% recall** is particularly important for an early-warning application because the system prioritizes detecting potentially geoeffective events.

---

## 🏗️ System Architecture

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

---

## 📁 Repository Structure

```text
ICME-Geoeffectiveness-Prediction/
│
├── cme_project.py       # Data processing and ML pipeline
├── app.py               # Application / prediction interface
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
├── data/
│   └── README.md        # Dataset information and sources
│
├── models/
│   └── README.md        # Trained model information
│
└── results/
    └── README.md        # Model evaluation results
```

> Large raw datasets are not included directly in this repository.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ICME-Geoeffectiveness-Prediction.git
cd ICME-Geoeffectiveness-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the main application:

```bash
python app.py
```

The application provides an interface for interacting with the trained prediction system.

---

## 🌍 Potential Impact

Reliable geomagnetic-storm prediction can support the resilience of:

* 🛰️ Satellite systems
* 📡 Communication infrastructure
* 🧭 Navigation and GNSS systems
* ⚡ Power-grid infrastructure
* 🚀 Space missions
* 🏭 Critical technological infrastructure

For India, developing indigenous AI-driven space-weather capabilities can contribute to **technological sovereignty and resilience as the country's satellite, navigation, communication, and commercial space ecosystem expands**.

The long-term vision is to evolve the system into an automated space-weather intelligence platform capable of:

```text
Real-Time Solar-Wind Data
          ↓
Automatic Data Processing
          ↓
AI Prediction
          ↓
GenAI-Based Explanation
          ↓
Automated Risk Assessment
          ↓
Early Warning / Alerts
```

---

## 🔬 Future Development

Planned improvements include:

* Automated ingestion of real-time solar-wind observations
* Continuous model validation with newly observed ICMEs
* Improved model calibration and generalization
* Generative AI explanations of model predictions
* Automated prediction and alert generation
* Deployment as a real-time space-weather monitoring service

---

## ⚠️ Disclaimer

This project is a **research prototype** and is not currently intended to replace operational space-weather forecasting systems.

The reported performance is based on historical data and should be further validated on independent, real-time observations before operational deployment.

---

## 👥 Project

Developed as an AI/ML-based approach to **ICME geoeffectiveness prediction and early geomagnetic-storm warning**.

**Goal:**
Build an indigenous, data-driven intelligence layer that can help improve preparedness against space-weather events.

---
