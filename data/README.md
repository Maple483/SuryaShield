## Data

The `data/` directory contains the raw and processed datasets used in the Suryashield pipeline.

### Source Data
- **Richardson & Cane ICME Catalog** — ICME event identification and timing.
- **NASA OMNI Dataset** — Time-series solar-wind and magnetic-field measurements.
- **Dst Index** — Geomagnetic storm intensity used for labeling ICME geoeffectiveness.

### Processed Data
The raw datasets are synchronized using ICME shock arrival times. For each event, the **first 3 hours of sequential solar-wind observations** are extracted and combined with the corresponding geoeffectiveness label to form the LSTM input dataset.

The final processed data is used for model training, validation, and testing.
