import pandas as pd

omni_path = "omni_1996_2015.csv"
richardson_path = "Richardson_catalog.csv"
output_path = "omni_icme_1996_2015_3hr.csv"

omni = pd.read_csv(omni_path)
rich = pd.read_csv(richardson_path)

# CLEAN RICHARDSON CATALOG

rich = pd.read_csv(
   "Richardson_catalog.csv"
)

shock_col = "Disturbance Y/M/D (UT) (a)"
start_col = "ICME Plasma/Field Start, End Y/M/D (UT) (b)"
end_col = "ICME Plasma/Field Start, End Y/M/D (UT) (b)_1"

# Convert all time columns to datetime

for col in [shock_col, start_col, end_col]:
    rich[col] = (
        rich[col]
        .astype(str)
        .str.replace(r"\s*\(.*?\)", "", regex=True)
        .str.strip()
    )
    rich[col] = pd.to_datetime(
        rich[col],
        format="%Y/%m/%d %H%M",
        errors="coerce"
    )

rich.rename(
    columns={
        shock_col: "Shock_Time",
        start_col: "ICME_Start",
        end_col: "ICME_End"
    },
    inplace=True
)

# Convert Dst to numeric

rich["Dst (nT) (m)"] = pd.to_numeric(
    rich["Dst (nT) (m)"],
    errors="coerce"
)

# Keep only events from 1996–2015
rich = rich[
    (rich["Shock_Time"].dt.year >= 1996) &
    (rich["Shock_Time"].dt.year <= 2015)
].reset_index(drop=True)

# Create Event_ID

rich["Event_ID"] = rich.index + 1

# Create Label

rich["Label"] = (
    rich["Dst (nT) (m)"] <= -50
).astype(int)

# Round shock time down to nearest hour

rich["Shock_Hour"] = rich["Shock_Time"].dt.floor("h")

# Save cleaned Richardson catalog
output = "Richardson_cleaned.csv"

rich.to_csv(
    output,
    index=False
)

# Convert OMNI datetime

omni["Datetime"] = pd.to_datetime(omni["Datetime"])

# Extract first 3 hours

events = []
for idx, row in rich.iterrows():
    shock = row["Shock_Hour"]
    event = omni[
        (omni["Datetime"] >= shock) &
        (omni["Datetime"] < shock + pd.Timedelta(hours=3))
    ].copy()
    if len(event) == 3:
        event["Event_ID"] = idx + 1
        event["Shock_Time"] = row["Shock_Time"]
        events.append(event)

if len(events) == 0:
    print("No events were extracted. Check your datetime formats.")
else:
    final_df = pd.concat(events, ignore_index=True)
    final_df.to_csv(
        output_path,
        index=False
    )

# 3. CREATE MASTER DATASET

rich = rich[
    [
        "Event_ID",
        "ICME_Start",
        "ICME_End",
        "Dst (nT) (m)",
        "Label"
    ]
]

# Merge

master = omni.merge(
    rich,
    on="Event_ID",
    how="left"
)
master.to_csv(
    output_path,
    index=False
)

# 4. DATA CLEANING

master = pd.read_csv(
    "Master_Dataset.csv",
    parse_dates=[
        "Datetime",
        "Shock_Time",
        "ICME_Start",
        "ICME_End"
    ]
)

omni_features = [
    "Bmag",
    "Bx",
    "By_GSM",
    "Bz_GSM",
    "Temperature",
    "Density",
    "Speed",
    "Pressure",
    "Ey",
    "Beta",
    "Mach",
    "Dst (nT) (m)"
]

for col in omni_features:
    master[col] = master[col].fillna(
        master[col].median()
    )

# Save cleaned master dataset

master.to_csv(
    "Master_Dataset_Cleaned.csv",
    index=False
)
