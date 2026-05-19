import os
import pandas as pd
import numpy as np
from io import StringIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, recall_score, accuracy_score, make_scorer

import matplotlib.pyplot as plt

from music21 import key as keyModule

# ========================
# LOAD DATA
# ========================

datasetDirectories = ['major', 'minor']

all_data = []

for datasetDir in datasetDirectories:

    for filename in os.listdir(datasetDir):

        if filename.endswith(".csv"):

            filepath = os.path.join(datasetDir, filename)

            with open(filepath, 'r') as f:
                lines = f.readlines()

            if len(lines) < 3:
                continue

            header = lines[0].strip().split(',')

            data_str = ''.join(lines[1:])

            df = pd.read_csv(StringIO(data_str), names=header)

            all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print(f"Total rows: {len(data)}")

# ========================
# FEATURES
# ========================

X = data.drop(columns=["targetChord"])

# one-hot encoding
X = pd.get_dummies(X)

y = data["targetChord"]

le = LabelEncoder()

y = le.fit_transform(y)

# ========================
# TRAIN TEST SPLIT
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ========================
# HELPERS
# ========================

def chord_to_pitch_set(label, key_obj):

    degree_str, quality = label.split('_')

    try:
        degree = int(degree_str)
    except:
        return set()

    root_pitch = key_obj.getScale().pitchFromDegree(degree)

    root_pc = root_pitch.pitchClass

    if quality == 'major':
        intervals = [0, 4, 7]

    elif quality == 'minor':
        intervals = [0, 3, 7]

    elif quality == 'diminished':
        intervals = [0, 3, 6]

    elif quality == 'augmented':
        intervals = [0, 4, 8]

    else:
        intervals = [0, 4, 7]

    return set((root_pc + i) % 12 for i in intervals)

def jaccard_distance(set1, set2):

    intersection = len(set1 & set2)

    return 3 - intersection

def get_func(label):

    degree = label.split('_')[0]

    tonic = {'1', '6'}
    subdominant = {'2', '4'}
    dominant = {'5', '7'}

    if degree in tonic:
        return 'T'

    if degree in subdominant:
        return 'S'

    if degree in dominant:
        return 'D'

    return 'X'

def evaluate_predictions(y_true, y_pred, le, key_str='C major'):

    y_true_labels = le.inverse_transform(y_true)

    y_pred_labels = le.inverse_transform(y_pred)

    total = len(y_true_labels)

    tonic, mode = key_str.split()

    key_obj = keyModule.Key(tonic, mode)

    f1_macro = f1_score(y_true, y_pred, average='macro')

    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    func_errors = sum(
        get_func(t) != get_func(p)
        for t, p in zip(y_true_labels, y_pred_labels)
    )

    func_mismatch_pct = func_errors / total

    jaccard_dists = [
        jaccard_distance(
            chord_to_pitch_set(t, key_obj),
            chord_to_pitch_set(p, key_obj)
        )
        for t, p in zip(y_true_labels, y_pred_labels)
    ]

    avg_jaccard = np.mean(jaccard_dists)

    return f1_macro, func_mismatch_pct, avg_jaccard, recall, accuracy

# ========================
# GRID SEARCH
# ========================
def functionMismatch(y_true, y_pred):

    y_true_labels = le.inverse_transform(y_true)
    y_pred_labels = le.inverse_transform(y_pred)

    total = len(y_true_labels)

    errors = sum(
        get_func(t) != get_func(p)
        for t, p in zip(y_true_labels, y_pred_labels)
    )

    return errors / total
def jaccardMetric(y_true, y_pred):

    y_true_labels = le.inverse_transform(y_true)
    y_pred_labels = le.inverse_transform(y_pred)

    tonic, mode = "C major".split()
    key_obj = keyModule.Key(tonic, mode)

    distances = [
        jaccard_distance(
            chord_to_pitch_set(t, key_obj),
            chord_to_pitch_set(p, key_obj)
        )
        for t, p in zip(y_true_labels, y_pred_labels)
    ]

    return np.mean(distances)
scoring = {

    "accuracy": make_scorer(accuracy_score),

    "funcMismatch": make_scorer(functionMismatch, greater_is_better=True),

    "jaccard": make_scorer(jaccardMetric, greater_is_better=True), 
    "f1_macro": "f1_macro"
}
param_grid = {

    'max_features': [4, 6, 8, 10],

    'n_estimators': [500, 800],

    'max_depth': [6, 8, 10, 12],

    'min_samples_split': [3],

    'min_samples_leaf': [2, 4, 6],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring=scoring,
    refit='accuracy'
)

grid.fit(X_train, y_train)

# ========================
# BEST MODEL
# ========================

print("\nBEST PARAMETERS:")
print(grid.best_params_)

print("\nBEST CV SCORE:")
print(grid.best_score_)

best_model = grid.best_estimator_

# ========================
# TEST EVALUATION
# ========================

y_pred = best_model.predict(X_test)

f1, func_err, jacc, recall, accuracy = evaluate_predictions(
    y_test,
    y_pred,
    le
)

print("\nTEST RESULTS")

print(f"F1 Macro: {f1:.4f}")

print(f"Recall Macro: {recall:.4f}")

print(f"Function mismatch: {func_err:.4f}")

print(f"Avg Jaccard distance: {jacc:.4f}")

# ========================
# FEATURE IMPORTANCE
# ========================

importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({

    'feature': X.columns,
    'importance': importances

}).sort_values(by='importance', ascending=False)

print("\nTOP 10 FEATURES:")

print(feature_importance_df.head(10))

# ========================
# RESULTS DATAFRAME
# ========================

results = pd.DataFrame(grid.cv_results_)

# ========================
# PLOT 1
# MAX FEATURES
# ========================

grouped_max_depth = results.groupby('param_max_features')[[
    "mean_test_accuracy",
    "mean_test_f1_macro",
    "mean_test_funcMismatch",
    "mean_test_jaccard"
]].mean()

plt.figure(figsize=(8, 5))

plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_accuracy"], marker='o')
plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_f1_macro"], marker='o')
plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_funcMismatch"], marker='o')
plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_jaccard"], marker='o')

plt.legend([
    "accuracy",
    "f1_macro",
    "funcMismatch",
    "jaccard"
])

plt.show()

# ========================
# PLOT 2
# MAX DEPTH
# ========================

grouped_max_depth = results.groupby('param_max_depth')[[
    "mean_test_accuracy",
    "mean_test_f1_macro",
    "mean_test_funcMismatch",
    "mean_test_jaccard"
]].mean()

plt.figure(figsize=(8, 5))

plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_accuracy"], marker='o')
plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_f1_macro"], marker='o')
plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_funcMismatch"], marker='o')
plt.plot(grouped_max_depth.index, grouped_max_depth["mean_test_jaccard"], marker='o')

plt.legend([
    "accuracy",
    "f1_macro",
    "funcMismatch",
    "jaccard"
])

plt.show()
# ========================
# PLOT 3
# N ESTIMATORS
# ========================

grouped_estimators = results.groupby(
    'param_n_estimators'
)[["mean_test_accuracy",
   'mean_test_f1_macro',
    "mean_test_funcMismatch",
    "mean_test_jaccard"]].mean()

plt.figure(figsize=(8, 5))

plt.plot(
    grouped_estimators.index,
    grouped_estimators.values,
    marker='o'
)

plt.xlabel("n_estimators")

plt.ylabel("Mean F1 Macro")

plt.title("F1 Macro vs n_estimators")

plt.grid(True)

plt.show()

# ========================
# PLOT 4
# MIN SAMPLES SPLIT
# ========================

