import pandas as pd
import numpy as np
import random
import tensorflow as tf

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

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

master = pd.read_csv(
    r"C:\Users\Ishita\Desktop\CME_project\Dataset\Master_Dataset.csv"
)

master["Datetime"] = pd.to_datetime(
    master["Datetime"],
    errors="coerce"
)

features_all = [
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

features_no_bx = [
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

feature_sets = {
    "All 11 Features": features_all,
    "Without Bx": features_no_bx
}

event_labels = (
    master
    .groupby("Event_ID")["Label"]
    .first()
)

event_ids = event_labels.index.values

train_events, test_events = train_test_split(
    event_ids,
    test_size=0.20,
    random_state=SEED,
    stratify=event_labels.loc[event_ids]
)

train_events, val_events = train_test_split(
    train_events,
    test_size=0.20,
    random_state=SEED,
    stratify=event_labels.loc[train_events]
)

results = []

def create_sequences(df, features):
    X = []
    y = []

    for event_id, group in df.groupby("Event_ID"):
        group = group.sort_values("Datetime")

        if len(group) != 3:
            continue

        X.append(
            group[features].values
        )

        y.append(
            group["Label"].iloc[0]
        )

    return np.array(X), np.array(y)

for model_name, features in feature_sets.items():

    print("\n========================================")
    print(model_name)
    print("========================================")

    train_df = master[
        master["Event_ID"].isin(train_events)
    ].copy()

    val_df = master[
        master["Event_ID"].isin(val_events)
    ].copy()

    test_df = master[
        master["Event_ID"].isin(test_events)
    ].copy()

    train_df = train_df.sort_values(
        ["Event_ID", "Datetime"]
    )

    val_df = val_df.sort_values(
        ["Event_ID", "Datetime"]
    )

    test_df = test_df.sort_values(
        ["Event_ID", "Datetime"]
    )

    medians = {}

    for col in features:
        median = train_df[col].median()
        medians[col] = median

        train_df[col] = train_df[col].fillna(median)
        val_df[col] = val_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

    scaler = StandardScaler()

    train_df[features] = scaler.fit_transform(
        train_df[features]
    )

    val_df[features] = scaler.transform(
        val_df[features]
    )

    test_df[features] = scaler.transform(
        test_df[features]
    )

    X_train, y_train = create_sequences(
        train_df,
        features
    )

    X_val, y_val = create_sequences(
        val_df,
        features
    )

    X_test, y_test = create_sequences(
        test_df,
        features
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)   

    model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    SpatialDropout1D(0.2),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
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

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=0
)

val_prob = model.predict(
    X_val,
    verbose=0
).ravel()

test_prob = model.predict(
    X_test,
    verbose=0
).ravel()

thresholds = np.arange(
    0.20,
    0.51,
    0.01
)

best_threshold = 0.5
best_val_f1 = -1

for threshold in thresholds:
    val_pred = (
        val_prob >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        best_threshold = threshold

test_pred = (
    test_prob >= best_threshold
).astype(int)

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

cm = confusion_matrix(
    y_test,
    test_pred
)

print("Features:", len(features))
print("Best threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_val_f1, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Confusion Matrix:")
print(cm)

results.append({
    "Model": model_name,
    "Features": len(features),
    "Threshold": best_threshold,
    "Validation_F1": best_val_f1,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
})

results_df = pd.DataFrame(results)

print("\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(
    results_df.round(4).to_string(
        index=False
    )
)

output_path = (
    "LSTM_model_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nSaved comparison to:")
print(output_path)

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n")
print("=" * 60)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 60)

baseline_prob = model.predict(
    X_test,
    verbose=0
).ravel()

baseline_auc = roc_auc_score(
    y_test,
    baseline_prob
)

print(
    "\nBaseline ROC-AUC:",
    round(baseline_auc, 4)
)

importance_results = []

rng = np.random.RandomState(42)

for feature_index, feature_name in enumerate(features):

    auc_scores = []

    for repeat in range(10):

        X_permuted = X_test.copy()

        permutation = rng.permutation(
            X_permuted.shape[0]
        )

        X_permuted[:, :, feature_index] = (
            X_test[
                permutation,
                :,
                feature_index
            ]
        )

        permuted_prob = model.predict(
            X_permuted,
            verbose=0
        ).ravel()

        permuted_auc = roc_auc_score(
            y_test,
            permuted_prob
        )

        auc_scores.append(
            permuted_auc
        )

    mean_auc = np.mean(
        auc_scores
    )

    std_auc = np.std(
        auc_scores
    )

    importance = (
        baseline_auc -
        mean_auc
    )

    importance_results.append({
        "Feature": feature_name,
        "Baseline_ROC_AUC": baseline_auc,
        "Permuted_ROC_AUC": mean_auc,
        "Importance": importance,
        "Std": std_auc
    })

permutation_df = pd.DataFrame(
    importance_results
)

permutation_df = (
    permutation_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nFEATURE IMPORTANCE")
print("=" * 60)

print(
    permutation_df[
        [
            "Feature",
            "Importance",
            "Permuted_ROC_AUC",
            "Std"
        ]
    ].to_string(
        index=False
    )
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    permutation_df["Feature"][::-1],
    permutation_df["Importance"][::-1]
)

plt.xlabel(
    "Decrease in ROC-AUC after permutation"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "LSTM Permutation Feature Importance"
)

plt.tight_layout()

plt.show()

output_path = (
    "LSTM_permutation_importance.csv"
)

permutation_df.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved permutation importance to:"
)

print(
    output_path
)
