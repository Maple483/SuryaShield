## Data

The `data/` directory contains the raw and processed datasets used in the Suryashield pipeline.
### Timeline: 1996-2025
### Source Data
- **Richardson & Cane ICME Catalog** — ICME event identification and timing and minimum dst index.
(https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm)
- Fields: Disturbance Y/M/D (UT) | ICME Plasma/Field Start | ICME Plasma/Field End Y/M/D (UT) | Dst (nT) (m)

- **NASA OMNI Dataset** — Time-series solar-wind measurements.
(https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/)
download the .dat files from 1996 to 2025
- Fields: Event_ID | Datetime	| Bmag	| Bx |	By_GSM | Bz_GSM |	Temperature |	Density |	Speed |	Pressure | Ey |	Beta | Mach

### Processed Data
The raw datasets are synchronized using ICME shock arrival times. For each event, the **first 3 hours of sequential solar-wind observations** are extracted and combined with the corresponding geoeffectiveness label to form the LSTM input dataset.

The final processed data is used for model training, validation, and testing.

## Inputs for NASA OMNI:
Create file
Select resolution -> Hourly averaged
