import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    SpatialDropout1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

st.set_page_config(
    page_title="ICME Geoeffectiveness Prediction",
    layout="wide"
)

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

NON_GEO_COLOR = "#0B3D91"
GEO_COLOR = "#F28E2B"

FEATURE_COLORS = {
    "Bmag": "#1F77B4",
    "Bx": "#9467BD",
    "By_GSM": "#2CA02C",
    "Bz_GSM": "#D62728",
    "Temperature": "#FF7F0E",
    "Density": "#17BECF",
    "Speed": "#E377C2",
    "Pressure": "#8C564B",
    "Ey": "#BCBD22",
    "Beta": "#7F7F7F",
    "Mach": "#4C78A8"
}

MASTER_PATH = (
    "Master_Dataset_1996_2025.csv"
)

FEATURES_11 = [
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
    "Mach"
]

FEATURES_10 = [
    "Bmag",
    "By_GSM",
    "Bz_GSM",
    "Temperature",
    "Density",
    "Speed",
    "Pressure",
    "Ey",
    "Beta",
    "Mach"
]

DISPLAY_NAMES = {
    "Bmag": "Bmag (nT)",
    "Bx": "Bx (nT)",
    "By_GSM": "By_GSM (nT)",
    "Bz_GSM": "Bz_GSM (nT)",
    "Temperature": "Temperature (K)",
    "Density": "Density (cm⁻³)",
    "Speed": "Speed (km/s)",
    "Pressure": "Pressure (nPa)",
    "Ey": "Ey (mV/m)",
    "Beta": "Beta",
    "Mach": "Mach"
}

@st.cache_data
def load_data():

    df = pd.read_csv(
        MASTER_PATH
    )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="coerce"
    )
    return df
master = load_data()

required_columns = [
    "Event_ID",
    "Datetime",
    "Label"
] + FEATURES_11

missing_columns = [
    col
    for col in required_columns
    if col not in master.columns
]

if missing_columns:
    st.error(
        "The following required columns are missing:"
    )
    st.write(
        missing_columns
    )
    st.stop()

def create_sequences(
    df,
    features
):
    X = []
    y = []
    event_ids = []
    for event_id, group in df.groupby(
        "Event_ID"
    ):
        group = group.sort_values(
            "Datetime"
        )
        if len(group) != 3:
            continue
        X.append(
            group[features].values
        )
        y.append(
            group["Label"].iloc[0]
        )
        event_ids.append(
            event_id
        )
    return (
        np.array(X),
        np.array(y),
        np.array(event_ids)
    )

@st.cache_resource
def train_lstm(
    feature_mode
):
    if feature_mode == "All 11 Features":
        features = FEATURES_11
    else:
        features = FEATURES_10

    event_labels = (
        master
        .groupby(
            "Event_ID"
        )["Label"]
        .first()
    )

    event_ids = (
        event_labels.index.values
    )

    train_events, test_events = train_test_split(
        event_ids,
        test_size=0.20,
        random_state=SEED,
        stratify=event_labels.loc[
            event_ids
        ]
    )

    train_events, val_events = train_test_split(
        train_events,
        test_size=0.20,
        random_state=SEED,
        stratify=event_labels.loc[
            train_events
        ]
    )

    train_df = master[
        master["Event_ID"].isin(
            train_events
        )
    ].copy()
    
    val_df = master[
        master["Event_ID"].isin(
            val_events
        )
    ].copy()

    test_df = master[
        master["Event_ID"].isin(
            test_events
        )
    ].copy()

    train_df = train_df.sort_values(
        [
            "Event_ID",
            "Datetime"
        ]
    )

    val_df = val_df.sort_values(
        [
            "Event_ID",
            "Datetime"
        ]
    )

    test_df = test_df.sort_values(
        [
            "Event_ID",
            "Datetime"
        ]
    )

    medians = {}
    for col in features:
        median = train_df[
            col
        ].median()

        medians[col] = median
        train_df[col] = (
            train_df[col]
            .fillna(median)
        )

        val_df[col] = (
            val_df[col]
            .fillna(median)
        )

        test_df[col] = (
            test_df[col]
            .fillna(median)
        )
        
    scaler = StandardScaler()
    train_df[features] = (
        scaler.fit_transform(
            train_df[features]
        )
    )

    val_df[features] = (
        scaler.transform(
            val_df[features]
        )
    )

    test_df[features] = (
        scaler.transform(
            test_df[features]
        )
    )

    X_train, y_train, train_ids = create_sequences(
        train_df,
        features
    )

    X_val, y_val, val_ids = create_sequences(
        val_df,
        features
    )

    X_test, y_test, test_ids = create_sequences(
        test_df,
        features
    )

    model = Sequential([
        Input(
            shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        ),

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            )
        ),

        SpatialDropout1D(
            0.2
        ),

        LSTM(
            32
        ),

        Dropout(
            0.3
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(
            0.2
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=150,
        batch_size=16,
        callbacks=[
            early_stopping,
            reduce_lr
        ],
        verbose=0
    )

   val_prob = model.predict(X_val, verbose=0).ravel()

test_prob = model.predict(X_test, verbose=0).ravel()

thresholds = np.arange(0.20, 0.51, 0.01)

best_threshold = 0.50
best_val_f1 = -1

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)
    current_f1 = f1_score(y_val, val_pred, zero_division=0)

    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        best_threshold = threshold

test_pred = (test_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, test_prob)
cm = confusion_matrix(y_test, test_pred)

return {
    "model": model,
    "scaler": scaler,
    "medians": medians,
    "features": features,
    "history": history,
    "best_threshold": best_threshold,
    "val_f1": best_val_f1,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "cm": cm,
    "test_ids": test_ids,
    "test_prob": test_prob,
    "test_pred": test_pred,
    "y_test": y_test
}

st.sidebar.title("ICME Prediction System")

section = st.sidebar.radio(
    "Go to",
    [
        "Visualization",
        "Prediction"
    ]
)

st.sidebar.markdown("---")

feature_mode = st.sidebar.radio(
    "LSTM Feature Set",
    [
        "Without Bx",
        "All 11 Features"
    ]
)
st.title("ICME Geoeffectiveness Prediction System")

st.caption(
    "LSTM-based prediction using the first 3 hours "
    "of in-situ solar-wind observations after ICME shock arrival."
)

with st.spinner("Loading LSTM model..."):
    results = train_lstm(feature_mode)

if section == "Visualization":
    st.header("Dataset Visualization")

    st.markdown(
        """
        Explore the 1996–2025 ICME dataset used by the LSTM model.
        Each ICME contributes three hourly measurements corresponding
        to the first three hours after shock arrival.
        """
    )

    total_events = master["Event_ID"].nunique()
    total_rows = len(master)

    geoeffective = (
        master[master["Label"] == 1]["Event_ID"].nunique()
    )

    non_geoeffective = (
        master[master["Label"] == 0]["Event_ID"].nunique()
    )

    valid_dates = master["Datetime"].dropna()

    if len(valid_dates) > 0:
        min_year = valid_dates.dt.year.min()
        max_year = valid_dates.dt.year.max()
    else:
        min_year = "N/A"
        max_year = "N/A"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total ICMEs", total_events)

    with col2:
        st.metric("Total Measurements", total_rows)

    with col3:
        st.metric("Geoeffective", geoeffective)

    with col4:
        st.metric("Dataset Coverage", f"{min_year}–{max_year}")

    st.subheader("ICME Class Distribution")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[
                "Non-Geoeffective (0)",
                "Geoeffective (1)"
            ],
            y=[
                non_geoeffective,
                geoeffective
            ],
            marker_color=[
                NON_GEO_COLOR,
                GEO_COLOR
            ],
            text=[
                non_geoeffective,
                geoeffective
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        title="Geoeffective vs Non-Geoeffective ICMEs",
        xaxis_title="ICME Class",
        yaxis_title="Number of ICMEs",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ICMEs by Year")

    year_df = (
        master
        .dropna(subset=["Datetime"])
        .copy()
    )

    year_df["Year"] = year_df["Datetime"].dt.year

    year_df = (
        year_df
        .groupby(["Year", "Label"])["Event_ID"]
        .nunique()
        .reset_index()
    )

    non_geo_year = year_df[
        year_df["Label"] == 0
    ]
geo_year = year_df[year_df["Label"] == 1]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=non_geo_year["Year"],
        y=non_geo_year["Event_ID"],
        name="Non-Geoeffective (0)",
        marker_color=NON_GEO_COLOR
    )
)

fig.add_trace(
    go.Bar(
        x=geo_year["Year"],
        y=geo_year["Event_ID"],
        name="Geoeffective (1)",
        marker_color=GEO_COLOR
    )
)

fig.update_layout(
    barmode="group",
    xaxis_title="Year",
    yaxis_title="Number of ICMEs"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Explore an Individual ICME")

st.write("Select the date of the ICME you want to explore.")

available_dates = (
    master["Datetime"]
    .dropna()
    .dt.date
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if available_dates:
    selected_date = st.selectbox("ICME Date", available_dates)

    date_events = (
        master[
            master["Datetime"].dt.date == selected_date
        ]["Event_ID"]
        .dropna()
        .unique()
    )

    if len(date_events) == 0:
        st.warning("No ICME event found for this date.")
        st.stop()

    if len(date_events) > 1:
        st.info(
            f"{len(date_events)} ICME events occur on this date."
        )

        selected_event = st.selectbox(
            "Select event on this date",
            date_events
        )
    else:
        selected_event = date_events[0]

    event_data = (
        master[
            master["Event_ID"] == selected_event
        ]
        .sort_values("Datetime")
        .copy()
    )

    event_label = int(event_data["Label"].iloc[0])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ICME Date", str(selected_date))

    with col2:
        st.metric("Measurements", len(event_data))

    with col3:
        st.metric(
            "Actual Label",
            (
                "Geoeffective (1)"
                if event_label == 1
                else "Non-Geoeffective (0)"
            )
        )

    st.subheader("ICME Measurements")

    st.dataframe(
        event_data,
        use_container_width=True
    )

    st.subheader("First 3 Hours of Solar-Wind Parameters")

    plot_features = [
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
        "Mach"
    ]

    selected_features = st.multiselect(
        "Select parameters to visualize",
        plot_features,
        default=[
            "Bmag",
            "Bz_GSM",
            "Speed",
            "Density"
        ]
    )

    if selected_features:
        plot_df = (
            event_data[
                ["Datetime"] + selected_features
            ]
            .melt(
                id_vars="Datetime",
                var_name="Feature",
                value_name="Value"
            )
        )

        fig = px.line(
            plot_df,
            x="Datetime",
            y="Value",
            color="Feature",
            markers=True,
            title="3-Hour Solar-Wind Evolution",
            color_discrete_map=FEATURE_COLORS
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.subheader("Feature Distribution")

distribution_feature = st.selectbox(
    "Select feature",
    FEATURES_11
)

distribution_df = master[
    [
        distribution_feature,
        "Label"
    ]
].copy()

distribution_df["Class"] = distribution_df["Label"].map({
    0: "Non-Geoeffective (0)",
    1: "Geoeffective (1)"
})

fig = go.Figure()

for label, color in [
    ("Non-Geoeffective (0)", NON_GEO_COLOR),
    ("Geoeffective (1)", GEO_COLOR)
]:
    values = (
        distribution_df[
            distribution_df["Class"] == label
        ][distribution_feature]
        .dropna()
    )

    fig.add_trace(
        go.Box(
            y=values,
            name=label,
            marker_color=color,
            line_color=color
        )
    )

fig.update_layout(
    title=distribution_feature,
    yaxis_title=distribution_feature,
    xaxis_title="ICME Class"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Feature Importance")

st.write(
    "Permutation importance shows how much the model's "
    "ROC-AUC changes when each feature is randomly "
    "permuted. Larger importance indicates greater "
    "reliance on that feature."
)
importance_11 = pd.DataFrame({
    "Feature": [
        "Density",
        "Bmag",
        "Temperature",
        "Speed",
        "Pressure",
        "Bz_GSM",
        "Ey",
        "By_GSM",
        "Bx",
        "Beta",
        "Mach"
    ],
    "Importance": [
        0.054755,
        0.025662,
        0.025098,
        0.023231,
        0.016891,
        0.011463,
        0.009683,
        0.001303,
        0.000000,
        -0.000217,
        -0.004299
    ]
})

importance_10 = pd.DataFrame({
    "Feature": [
        "Density",
        "Bmag",
        "Temperature",
        "Speed",
        "Pressure",
        "Bz_GSM",
        "Ey",
        "By_GSM",
        "Beta",
        "Mach"
    ],
    "Importance": [
        0.054755,
        0.025662,
        0.025098,
        0.023231,
        0.016891,
        0.011463,
        0.009683,
        0.001303,
        -0.000217,
        -0.004299
    ]
})

importance_option = st.radio(
    "Feature set",
    [
        "All 11 Features",
        "Without Bx (10 Features)"
    ],
    horizontal=True,
    key="importance_feature_set"
)

if importance_option == "All 11 Features":
    importance_df = importance_11.copy()
else:
    importance_df = importance_10.copy()

importance_df = (
    importance_df
    .sort_values("Importance", ascending=True)
    .reset_index(drop=True)
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker_color=[
            FEATURE_COLORS.get(feature, "#7B2CBF")
            for feature in importance_df["Feature"]
        ],
        text=[
            f"{value:.4f}"
            for value in importance_df["Importance"]
        ],
        textposition="outside"
    )
)

fig.update_layout(
    title=(
        "Permutation Feature Importance — "
        + importance_option
    ),
    xaxis_title="Decrease in ROC-AUC after permutation",
    yaxis_title="Feature",
    height=550,
    showlegend=False,
    margin=dict(
        l=100,
        r=80,
        t=80,
        b=60
    )
)

fig.add_vline(
    x=0,
    line_width=1,
    line_dash="dash",
    line_color="black"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "Interpretation: a positive permutation importance means "
    "that randomly disrupting the feature reduced model ROC-AUC, "
    "suggesting that the model relies on that feature. Values close "
    "to zero indicate little additional predictive contribution, "
    "while negative values can occur when permutation happens to "
    "improve performance because of sampling variability or "
    "feature redundancy."
)

st.header("LSTM Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        f"{results['accuracy']:.3f}"
    )

with col2:
    st.metric(
        "Precision",
        f"{results['precision']:.3f}"
    )

with col3:
    st.metric(
        "Recall",
        f"{results['recall']:.3f}"
    )

with col4:
    st.metric(
        "F1",
        f"{results['f1']:.3f}"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{results['roc_auc']:.3f}"
    )

st.write(
    f"Classification threshold: "
    f"**{results['best_threshold']:.2f}**"
)

cm = results["cm"]

fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=[
            "Predicted Non-Geoeffective",
            "Predicted Geoeffective"
        ],
        y=[
            "Actual Non-Geoeffective",
            "Actual Geoeffective"
        ],
        text=cm,
        texttemplate="%{text}",
        colorscale=[
            [0, "#F3E8FF"],
            [0.5, "#A855F7"],
            [1, "#6B21A8"]
        ]
    )
)

fig.update_layout(title="Confusion Matrix")

st.plotly_chart(
    fig,
    use_container_width=True
)

if section == "Prediction":
    st.header("ICME Geoeffectiveness Prediction")

    st.markdown(
        """
        Enter the solar-wind measurements for the **first 3 hours
        after ICME shock arrival**.

        The LSTM processes these three measurements as a time
        sequence and estimates the probability that the ICME
        will be geoeffective.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Features",
            len(results["features"])
        )

    with col2:
        st.metric(
            "Sequence Length",
            "3 hours"
        )

    with col3:
        st.metric(
            "Threshold",
            f"{results['best_threshold']:.2f}"
        )

    st.markdown("---")

    st.subheader("Enter Solar-Wind Measurements")

    st.caption(
        "Enter the measurements for each of the first 3 hours "
        "after ICME shock arrival. Units are shown in brackets "
        "in the column headers."
    )

    features = results["features"]

    default_values = {}

    for feature in features:
        default_values[feature] = results["medians"][feature]

    input_rows = []

    for hour in range(3):
        row = {"Time Step": hour}

        for feature in features:
            row[feature] = float(default_values[feature])

        input_rows.append(row)

    input_df = pd.DataFrame(input_rows)

    display_df = input_df.copy()

    display_columns = {
        "Time Step": "Time Step",
        **{
            feature: DISPLAY_NAMES[feature]
            for feature in features
        }
    }

    display_df = display_df.rename(
        columns=display_columns
    )

    edited_display_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Time Step"],
        key="prediction_input"
    )

    reverse_display_columns = {
        DISPLAY_NAMES[feature]: feature
        for feature in features
    }

    edited_df = edited_display_df.rename(
        columns=reverse_display_columns
    )

    st.markdown("---")

    predict_button = st.button(
        "Predict Geoeffectiveness",
        type="primary",
        use_container_width=True
    )

    if predict_button:
        try:
            prediction_data = (
                edited_df[features]
                .astype(float)
            )

            scaled_data = (
                results["scaler"]
                .transform(prediction_data)
            )

            X_new = (
                scaled_data
                .reshape(
                    1,
                    3,
                    len(features)
                )
            )

            probability = float(
                results["model"]
                .predict(
                    X_new,
                    verbose=0
                )[0][0]
            )

            threshold = results["best_threshold"]

            prediction = (
                1
                if probability >= threshold
                else 0
            )
fig.update_layout(title="Confusion Matrix")

st.plotly_chart(
    fig,
    use_container_width=True
)

if section == "Prediction":
    st.header("ICME Geoeffectiveness Prediction")

    st.markdown(
        """
        Enter the solar-wind measurements for the **first 3 hours
        after ICME shock arrival**.

        The LSTM processes these three measurements as a time
        sequence and estimates the probability that the ICME
        will be geoeffective.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Features", len(results["features"]))

    with col2:
        st.metric("Sequence Length", "3 hours")

    with col3:
        st.metric(
            "Threshold",
            f"{results['best_threshold']:.2f}"
        )

    st.markdown("---")

    st.subheader("Enter Solar-Wind Measurements")

    st.caption(
        "Enter the measurements for each of the first 3 hours "
        "after ICME shock arrival. Units are shown in brackets "
        "in the column headers."
    )

    features = results["features"]

    default_values = {}

    for feature in features:
        default_values[feature] = results["medians"][feature]

    input_rows = []

    for hour in range(3):
        row = {"Time Step": hour}

        for feature in features:
            row[feature] = float(default_values[feature])

        input_rows.append(row)

    input_df = pd.DataFrame(input_rows)

    display_df = input_df.copy()

    display_columns = {
        "Time Step": "Time Step",
        **{
            feature: DISPLAY_NAMES[feature]
            for feature in features
        }
    }

    display_df = display_df.rename(
        columns=display_columns
    )

    edited_display_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Time Step"],
        key="prediction_input"
    )

    reverse_display_columns = {
        DISPLAY_NAMES[feature]: feature
        for feature in features
    }

    edited_df = edited_display_df.rename(
        columns=reverse_display_columns
    )

    st.markdown("---")

    predict_button = st.button(
        "Predict Geoeffectiveness",
        type="primary",
        use_container_width=True
    )

    if predict_button:
        try:
            prediction_data = (
                edited_df[features]
                .astype(float)
            )

            scaled_data = (
                results["scaler"]
                .transform(prediction_data)
            )

            X_new = scaled_data.reshape(
                1,
                3,
                len(features)
            )

            probability = float(
                results["model"]
                .predict(
                    X_new,
                    verbose=0
                )[0][0]
            )

            threshold = results["best_threshold"]

            prediction = (
                1
                if probability >= threshold
                else 0
            )

            st.markdown("---")

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("GEOEFFECTIVE ICME")
                st.write(
                    "The model predicts that this ICME "
                    "is likely to produce a significant "
                    "geomagnetic disturbance."
                )
            else:
                st.success("NON-GEOEFFECTIVE ICME")
                st.write(
                    "The model predicts that this ICME "
                    "is unlikely to produce a significant "
                    "geomagnetic disturbance."
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Geoeffectiveness Probability",
                    f"{probability * 100:.1f}%"
                )

            with col2:
                st.metric(
                    "Decision Threshold",
                    f"{threshold * 100:.1f}%"
                )

            with col3:
                st.metric(
                    "Prediction",
                    (
                        "Geoeffective (1)"
                        if prediction == 1
                        else "Non-Geoeffective (0)"
                    )
                )

            st.subheader("Probability of Geoeffectiveness")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 42
                        }
                    },
                    title={
                        "text": "Probability of Geoeffective ICME"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickmode": "array",
                            "tickvals": [
                                0,
                                20,
                                40,
                                60,
                                80,
                                100
                            ],
                            "ticktext": [
                                "0%",
                                "20%",
                                "40%",
                                "60%",
                                "80%",
                                "100%"
                            ]
                        },
                        "steps": [
                            {
                                "range": [0, 20],
                                "color": "#DCEBFF"
                            },
                            {
                                "range": [20, 40],
                                "color": "#A9D6FF"
                            },
                            {
                                "range": [40, 60],
                                "color": "#FFE6A7"
                            },
                            {
                                "range": [60, 80],
                                "color": "#FFB870"
                            },
                            {
                                "range": [80, 100],
                                "color": "#F28E2B"
                            }
                        ],
                        "bar": {
                            "color": "#333333"
                        },
                        "threshold": {
                            "line": {
                                "color": "#6B21A8",
                                "width": 5
                            },
                            "thickness": 0.8,
                            "value": threshold * 100
                        }
                    }
                )
            )

            gauge.update_layout(
                height=400,
                margin=dict(
                    l=30,
                    r=30,
                    t=80,
                    b=30
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.caption(
                f"Model decision threshold: "
                f"{threshold * 100:.1f}%"
            )

            if probability < 0.20:
                st.info(
                    "Very low predicted probability of geoeffectiveness."
                )
            elif probability < 0.40:
                st.info(
                    "Low predicted probability of geoeffectiveness."
                )
            elif probability < 0.60:
                st.warning(
                    "Moderate predicted probability of geoeffectiveness."
                )
            elif probability < 0.80:
                st.warning(
                    "High predicted probability of geoeffectiveness."
                )
            else:
                st.error(
                    "Very high predicted probability of geoeffectiveness."
                )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

st.markdown("---")

st.caption(
    """
    ICME Geoeffectiveness Prediction System |
    Richardson & Cane + NASA OMNI |
    1996–2025 |
    First 3 hours after shock arrival |
    LSTM |
    Random Seed = 42
    """
)

def explain_prediction(
    model,
    scaler,
    input_data,
    features,
    original_probability,
    n_repeats=10
):
    results = []

    scaled_original = scaler.transform(
        input_data[features]
    )

    X_original = scaled_original.reshape(
        1,
        3,
        len(features)
    )

    for feature_index, feature in enumerate(features):
        probability_changes = []

        for _ in range(n_repeats):
            X_perturbed = X_original.copy()

            original_values = (
                X_perturbed[
                    0,
                    :,
                    feature_index
                ].copy()
            )

            shuffled_values = original_values.copy()

            np.random.shuffle(shuffled_values)

            X_perturbed[
                0,
                :,
                feature_index
            ] = shuffled_values

            perturbed_probability = float(
                model.predict(
                    X_perturbed,
                    verbose=0
                )[0][0]
            )

            change = (
                perturbed_probability
                - original_probability
            )

            probability_changes.append(change)

        mean_change = np.mean(
            probability_changes
        )

        mean_absolute_change = np.mean(
            np.abs(probability_changes)
        )

        results.append({
            "Feature": feature,
            "Probability_Change": mean_change,
            "Importance": mean_absolute_change
        })

    explanation_df = pd.DataFrame(results)

    explanation_df = (
        explanation_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return explanation_df
