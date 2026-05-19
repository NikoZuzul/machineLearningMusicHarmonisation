import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

# ========================
# 1. UČITAVANJE PODATAKA IZ FOLDERA major i minor
# ========================

def load_data_from_folders(folders):
    all_data = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder, filename)
                df = pd.read_csv(filepath)
                all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)
    return data

folders = ['major', 'minor']
data = load_data_from_folders(folders)
print(f"Učitano redaka: {len(data)}")

# ========================
# 2. DODAVANJE BUDUĆIH ZNAČAJKI (3 sljedeće note)
# ========================

note_base_features = [
    "scaleDegree",
    "beatStrength",
    "duration",
    "beat",
    "measure"
]

note_ids = [0, 1, 2, 3]
future_steps = 3

for f in note_base_features:
    for n in note_ids:
        col = f"note{n}_{f}"
        for i in range(1, future_steps + 1):
            data[f"{col}_future_{i}"] = data[col].shift(-i)

# Nakon pomicanja, redovi s NaN se uklanjaju
data = data.dropna().reset_index(drop=True)

# ========================
# 3. IZRAČUN INTERVALA IZMEĐU NOTA (trenutne note, bez budućih)
# ========================

data['interval_0_1'] = data['note1_scaleDegree'] - data['note0_scaleDegree']
data['interval_1_2'] = data['note2_scaleDegree'] - data['note1_scaleDegree']
data['interval_2_3'] = data['note3_scaleDegree'] - data['note2_scaleDegree']

data['abs_interval_0_1'] = data['interval_0_1'].abs()
data['abs_interval_1_2'] = data['interval_1_2'].abs()
data['abs_interval_2_3'] = data['interval_2_3'].abs()

# ========================
# 4. PRIPREMA PODATAKA ZA MODELE
# ========================

# Target i features
target = 'targetChord'

X = data.drop(columns=[target])
y = data[target]

# Kategoriziraj kolone koje su stringovi (npr. prevChord0, prevChord1, prevChord2)
cat_cols = [c for c in X.columns if X[c].dtype == 'object']

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Enkodiranje ciljne varijable
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Provjera distribucije klasa prije podjele
class_counts = pd.Series(y_enc).value_counts()
print("Raspodjela klasa prije filtriranja:", dict(zip(le.classes_, class_counts)))

# Filtriraj klase koje imaju manje od 2 primjera (sprječava grešku kod stratificiranog split-a)
min_samples = 2
valid_classes = class_counts[class_counts >= min_samples].index

mask = pd.Series(y_enc).isin(valid_classes)
X = X.loc[mask.values]
y_enc = y_enc[mask.values]

print("Raspodjela klasa nakon filtriranja:", pd.Series(y_enc).value_counts())

# Podjela podataka (stratificirano)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Standardizacija za logističku regresiju
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# 5. DEFINICIJA I TRENING MODELA
# ========================

# Random Forest
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [6, 8, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 0.5]
}

print("Pokrećem grid search za Random Forest...")
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

y_pred_rf = rf_best.predict(X_test)

print("\n=== Random Forest Rezultati ===")
print(f"Najbolji parametri: {rf_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1 macro: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")
print(f"Recall macro: {recall_score(y_test, y_pred_rf, average='macro'):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Logistička regresija
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced',
                        solver='lbfgs', multi_class='multinomial')

lr_param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

print("Pokrećem grid search za Logističku regresiju...")
lr_grid = GridSearchCV(lr, lr_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_

y_pred_lr = lr_best.predict(X_test_scaled)

print("\n=== Logistička regresija Rezultati ===")
print(f"Najbolji parametri: {lr_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"F1 macro: {f1_score(y_test, y_pred_lr, average='macro'):.4f}")
print(f"Recall macro: {recall_score(y_test, y_pred_lr, average='macro'):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# ========================
# 6. ANALIZA VAŽNOSTI ZNAČAJKI
# ========================

# Permutacijska važnost za Random Forest
print("Računam permutation importance za Random Forest...")
perm_imp = permutation_importance(rf_best, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

fi_df = pd.DataFrame({
    'feature': X.columns,
    'perm_importance_mean': perm_imp.importances_mean,
    'perm_importance_std': perm_imp.importances_std
}).sort_values('perm_importance_mean', ascending=False)

print("\nTop 20 značajki (permutation importance RF):")
print(fi_df.head(20).to_string(index=False))

# Koeficijenti značajki za logističku regresiju
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coef_mean_abs': np.abs(lr_best.coef_).mean(axis=0)
}).sort_values('coef_mean_abs', ascending=False)

print("\nTop 20 značajki po apsolutnoj vrijednosti koeficijenata (LR):")
print(coefficients.head(20).to_string(index=False))

# ========================
# 7. GRUPIRANJE ZNAČAJKI I VIZUALIZACIJA
# ========================

def feature_group(name):
    if '_future_' in name:
        return 'future_note'
    elif name.startswith('note'):
        return 'current_note'
    elif name.startswith('interval'):
        return 'interval'
    elif name.startswith('prevChord'):
        return 'prev_chord'
    else:
        return 'other'

fi_df['group'] = fi_df['feature'].apply(feature_group)
group_imp = fi_df.groupby('group')['perm_importance_mean'].sum().sort_values(ascending=False)

print("\nUkupna važnost grupa značajki (RF):")
print(group_imp)

# Graf: Top 20 značajki RF
plt.figure(figsize=(12,6))
plt.barh(fi_df.head(20)['feature'][::-1], fi_df.head(20)['perm_importance_mean'][::-1])
plt.title('Top 20 značajki (Random Forest permutation importance)')
plt.xlabel('Permutation importance')
plt.tight_layout()
plt.show()

# Graf: Top 20 značajki LR
plt.figure(figsize=(12,6))
plt.barh(coefficients.head(20)['feature'][::-1], coefficients.head(20)['coef_mean_abs'][::-1])
plt.title('Top 20 značajki (Logistička regresija - apsolutne vrijednosti koeficijenata)')
plt.xlabel('Apsolutna vrijednost koeficijenta')
plt.tight_layout()
plt.show()
