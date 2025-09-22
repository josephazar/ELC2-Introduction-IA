# 💼 TD: Prédiction de Salaires avec Régression Linéaire

## 📋 Objectif
Apprendre à construire un modèle de régression linéaire complet, de la préparation des données jusqu'à l'évaluation, en utilisant un dataset de salaires d'entreprise.

## 🎯 Ce que vous allez apprendre
- Préparer des données réelles avec tous leurs défauts
- Gérer les problèmes d'échelle et de format
- Construire différents modèles de régression (simple, multiple, polynomiale)
- Évaluer et comparer les performances
- Interpréter les résultats

## 🎭 Contexte
Vous êtes data scientist dans une entreprise de 200 employés. Le département RH vous demande de créer un modèle pour prédire les salaires en fonction de différents critères. Cela aidera à :
- Assurer l'équité salariale
- Prévoir le budget RH
- Comprendre les facteurs influençant les salaires

---

## 📚 Étape 0: Préparation de l'environnement

### Importation des bibliothèques

```python
# Bibliothèques essentielles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Pour la régression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Pour le prétraitement
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Pour l'évaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration visuelle
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

# Ignorer les warnings
import warnings
warnings.filterwarnings('ignore')
```

---

## 📊 Étape 1: Chargement et Exploration des Données

### 1.1 Chargement du dataset

```python
# Charger les données
df = pd.read_csv('data/salaires_entreprise.csv')

print("=== DATASET CHARGÉ ===")
print(f"Dimensions: {df.shape}")
print(f"Nombre d'employés: {df.shape[0]}")
print(f"Nombre de variables: {df.shape[1]}")
```

### 1.2 Première exploration

```python
# Afficher les premières lignes
print("\n=== APERÇU DES DONNÉES ===")
print(df.head())

# Information sur les colonnes
print("\n=== INFORMATIONS SUR LES COLONNES ===")
print(df.info())

# Statistiques descriptives
print("\n=== STATISTIQUES DESCRIPTIVES ===")
print(df.describe())
```

### 1.3 Comprendre chaque variable

```python
# Analyser chaque colonne
print("\n=== ANALYSE DES VARIABLES ===")

for col in df.columns:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Valeurs manquantes: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
    if df[col].dtype == 'object':
        print(f"  Valeurs uniques: {df[col].nunique()}")
        print(f"  Échantillon: {df[col].value_counts().head(3).to_dict()}")
    else:
        print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
```

### 📝 Questions à se poser:
1. **Quelle est la variable cible?** → `salary`
2. **Quelles variables semblent importantes?** → experience, education, department...
3. **Y a-t-il des problèmes évidents?** → Valeurs manquantes, formats mixtes, outliers...

---

## 🔍 Étape 2: Détection des Problèmes

### 2.1 Visualiser les distributions

```python
# Visualiser les variables numériques
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution de {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Fréquence')

        # Ajouter la médiane
        median = df[col].median()
        axes[idx].axvline(median, color='red', linestyle='--', label=f'Médiane: {median:.0f}')
        axes[idx].legend()

plt.tight_layout()
plt.show()
```

### 2.2 Identifier les problèmes spécifiques

```python
print("\n=== PROBLÈMES DÉTECTÉS ===")

# 1. Valeurs manquantes
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
print("\n1. VALEURS MANQUANTES:")
print(missing_df[missing_df['Count'] > 0])

# 2. Problèmes de format dans 'city'
print("\n2. PROBLÈME DE CASSE dans 'city':")
print(df['city'].value_counts())
print("→ Paris, paris, PARIS sont la même ville!")

# 3. Format mixte dans 'certification'
print("\n3. FORMAT MIXTE dans 'certification':")
print(df['certification'].value_counts())
print("→ Oui/OUI/Yes/1 sont identiques!")

# 4. Outliers potentiels
print("\n4. OUTLIERS POTENTIELS:")
for col in ['salary', 'performance_score', 'hours_per_week']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"  {col}: {len(outliers)} outliers détectés")

# 5. Échelles différentes
print("\n5. PROBLÈME D'ÉCHELLE dans 'satisfaction':")
print(f"  Min: {df['satisfaction'].min():.2f}")
print(f"  Max: {df['satisfaction'].max():.2f}")
print("  → Certaines valeurs sont sur 0-10, d'autres sur 0-100!")
```

---

## 🧹 Étape 3: Nettoyage et Prétraitement

### 3.1 Corriger les problèmes de format

```python
# Créer une copie pour travailler
df_clean = df.copy()

# 1. Harmoniser les villes (mettre en titre)
print("=== NETTOYAGE DES VILLES ===")
df_clean['city'] = df_clean['city'].str.title()  # Paris, paris, PARIS → Paris
print(f"Villes uniques après nettoyage: {df_clean['city'].nunique()}")

# 2. Harmoniser certification
print("\n=== NETTOYAGE CERTIFICATION ===")
certification_mapping = {
    'Oui': 1, 'OUI': 1, 'Yes': 1, '1': 1, 1: 1,
    'Non': 0, 'non': 0, 'No': 0, '0': 0, 0: 0
}
df_clean['certification'] = df_clean['certification'].map(lambda x: certification_mapping.get(x, np.nan))
print(f"Valeurs uniques: {df_clean['certification'].unique()}")

# 3. Corriger l'échelle de satisfaction (détecter et convertir les valeurs 0-10 en 0-100)
print("\n=== CORRECTION ÉCHELLE SATISFACTION ===")
# Les valeurs < 15 sont probablement sur une échelle 0-10
mask_small_scale = df_clean['satisfaction'] < 15
df_clean.loc[mask_small_scale, 'satisfaction'] = df_clean.loc[mask_small_scale, 'satisfaction'] * 10
print(f"Valeurs converties: {mask_small_scale.sum()}")
print(f"Nouvelle plage: [{df_clean['satisfaction'].min():.0f}, {df_clean['satisfaction'].max():.0f}]")

# 4. Gérer les outliers de performance_score
print("\n=== CORRECTION PERFORMANCE SCORE ===")
# Limiter entre 1 et 10
df_clean['performance_score'] = df_clean['performance_score'].clip(1, 10)
print(f"Performance score maintenant entre {df_clean['performance_score'].min()} et {df_clean['performance_score'].max()}")
```

### 3.2 Traiter les valeurs manquantes

```python
print("\n=== TRAITEMENT DES VALEURS MANQUANTES ===")

# Stratégies par colonne
# age: imputer avec la médiane
median_age = df_clean['age'].median()
df_clean['age'].fillna(median_age, inplace=True)
print(f"Age: imputé avec médiane = {median_age:.1f}")

# experience: imputer avec la médiane selon le département
for dept in df_clean['department'].unique():
    mask = df_clean['department'] == dept
    median_exp = df_clean.loc[mask, 'experience'].median()
    df_clean.loc[mask, 'experience'] = df_clean.loc[mask, 'experience'].fillna(median_exp)
print(f"Experience: imputé par département")

# education: remplacer par le mode (plus fréquent)
mode_education = df_clean['education'].mode()[0]
df_clean['education'].fillna(mode_education, inplace=True)
print(f"Education: imputé avec mode = {mode_education}")

# certification: imputer avec 0 (pas de certification)
df_clean['certification'].fillna(0, inplace=True)
print(f"Certification: valeurs manquantes → 0 (pas de certification)")

# city: remplacer par 'Unknown' si manquant
df_clean['city'].fillna('Unknown', inplace=True)
print(f"City: valeurs manquantes → 'Unknown'")

# Vérifier
print(f"\nValeurs manquantes restantes: {df_clean.isnull().sum().sum()}")
```

### 3.3 Feature Engineering

```python
print("\n=== CRÉATION DE NOUVELLES VARIABLES ===")

# 1. Ratio performance/heures
df_clean['efficiency'] = df_clean['performance_score'] / df_clean['hours_per_week'] * 40
print(f"✅ Efficiency créée (performance ajustée aux heures)")

# 2. Catégorie d'âge
df_clean['age_category'] = pd.cut(df_clean['age'],
                                   bins=[20, 30, 40, 50, 65],
                                   labels=['Junior', 'Confirmé', 'Senior', 'Expert'])
print(f"✅ Age_category créée")

# 3. Niveau d'expérience
df_clean['experience_level'] = pd.cut(df_clean['experience'],
                                       bins=[0, 3, 7, 15, 40],
                                       labels=['Débutant', 'Intermédiaire', 'Expérimenté', 'Expert'])
print(f"✅ Experience_level créée")

# 4. Grande ville (Paris, Lyon, Marseille)
df_clean['big_city'] = df_clean['city'].isin(['Paris', 'Lyon', 'Marseille']).astype(int)
print(f"✅ Big_city créée (1 si grande ville, 0 sinon)")

print(f"\nNombre total de variables: {df_clean.shape[1]}")
```

---

## 🏷️ Étape 4: Encodage des Variables Catégorielles

### 4.1 Identifier et encoder

```python
# Séparer les variables
categorical_cols = ['education', 'department', 'city', 'age_category', 'experience_level']
numerical_cols = ['age', 'experience', 'performance_score', 'nb_projects',
                  'hours_per_week', 'team_size', 'certification', 'satisfaction',
                  'efficiency', 'big_city']
target_col = 'salary'

print("=== ENCODAGE DES VARIABLES CATÉGORIELLES ===")

# Créer un dataframe pour stocker les features encodées
df_encoded = df_clean[numerical_cols].copy()

# 1. Encoder education (ordinal)
education_mapping = {'Bac': 0, 'Licence': 1, 'Master': 2, 'Doctorat': 3}
df_encoded['education_encoded'] = df_clean['education'].map(education_mapping)
print(f"✅ Education encodée (ordinal): {education_mapping}")

# 2. One-hot encoding pour department (nominal)
dept_dummies = pd.get_dummies(df_clean['department'], prefix='dept', drop_first=True)
df_encoded = pd.concat([df_encoded, dept_dummies], axis=1)
print(f"✅ Department encodé (one-hot): {dept_dummies.columns.tolist()}")

# 3. One-hot encoding pour les principales villes seulement
top_cities = df_clean['city'].value_counts().head(5).index.tolist()
for city in top_cities:
    if city != 'Unknown':
        df_encoded[f'city_{city}'] = (df_clean['city'] == city).astype(int)
print(f"✅ Top villes encodées: {top_cities}")

print(f"\nNombre de features après encodage: {df_encoded.shape[1]}")
```

---

## ⚖️ Étape 5: Analyse de Corrélation

### 5.1 Matrice de corrélation

```python
# Ajouter la variable cible
df_for_corr = pd.concat([df_encoded, df_clean[[target_col]]], axis=1)

# Calculer la corrélation
correlation_matrix = df_for_corr.corr()

# Visualiser
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Top corrélations avec le salaire
print("\n=== TOP CORRÉLATIONS AVEC LE SALAIRE ===")
salary_corr = correlation_matrix['salary'].abs().sort_values(ascending=False)
print(salary_corr.head(10))
```

### 5.2 Éliminer la multicolinéarité

```python
# Détecter les variables fortement corrélées entre elles
print("\n=== MULTICOLINÉARITÉ DÉTECTÉE ===")

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            if col1 != 'salary' and col2 != 'salary':
                high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))

for pair in high_corr_pairs:
    print(f"{pair[0]} ↔ {pair[1]}: {pair[2]:.2f}")

# Décider quelles variables supprimer
# Par exemple, si age et experience sont très corrélés, on peut garder seulement experience
if len(high_corr_pairs) > 0:
    print("\n→ Recommandation: Supprimer une variable de chaque paire fortement corrélée")
```

---

## 🔄 Étape 6: Division Train/Test

### 6.1 Préparer X et y

```python
# Supprimer les colonnes fortement corrélées si nécessaire
# (exemple: si age et experience sont trop corrélés)
columns_to_drop = []  # Ajouter les colonnes à supprimer selon l'analyse

X = df_encoded.drop(columns_to_drop, axis=1, errors='ignore')
y = df_clean[target_col]

print(f"=== DIMENSIONS DES DONNÉES ===")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Features: {X.columns.tolist()}")
```

### 6.2 Diviser en train/test

```python
# Division 80/20 avec stratification approximative
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n=== DIVISION TRAIN/TEST ===")
print(f"Train: {X_train.shape[0]} échantillons")
print(f"Test: {X_test.shape[0]} échantillons")
print(f"Ratio: {X_train.shape[0]/len(X)*100:.0f}% / {X_test.shape[0]/len(X)*100:.0f}%")
```

### 6.3 Normalisation

```python
# Créer et ajuster le scaler sur le train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # IMPORTANT: seulement transform

print("\n=== NORMALISATION ===")
print(f"Moyenne train (devrait être ~0): {X_train_scaled.mean():.6f}")
print(f"Écart-type train (devrait être ~1): {X_train_scaled.std():.6f}")

# Convertir en DataFrame pour garder les noms de colonnes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
```

---

## 📈 Étape 7: Construction des Modèles

### 7.1 Modèle de base (moyenne)

```python
print("=== MODÈLE DE BASE (BASELINE) ===")

# Prédire toujours la moyenne
y_pred_baseline = np.full(len(y_test), y_train.mean())

# Évaluation
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"RMSE: {rmse_baseline:.0f} €")
print(f"MAE: {mae_baseline:.0f} €")
print(f"R²: {r2_baseline:.3f}")
print("→ C'est notre référence à battre!")
```

### 7.2 Régression Linéaire Simple

```python
print("\n=== RÉGRESSION LINÉAIRE SIMPLE ===")

# Choisir la meilleure variable individuelle (celle avec la plus forte corrélation)
best_single_feature = salary_corr.index[1]  # index 0 est 'salary' lui-même
print(f"Meilleure feature individuelle: {best_single_feature}")

# Entraîner le modèle simple
model_simple = LinearRegression()
X_train_simple = X_train_scaled[[best_single_feature]]
X_test_simple = X_test_scaled[[best_single_feature]]

model_simple.fit(X_train_simple, y_train)

# Prédictions
y_pred_simple = model_simple.predict(X_test_simple)

# Évaluation
mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f"Équation: Salaire = {model_simple.coef_[0]:.0f} × {best_single_feature} + {model_simple.intercept_:.0f}")
print(f"RMSE: {rmse_simple:.0f} € (baseline: {rmse_baseline:.0f})")
print(f"MAE: {mae_simple:.0f} €")
print(f"R²: {r2_simple:.3f} (baseline: {r2_baseline:.3f})")

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test, alpha=0.5, label='Données réelles')
plt.scatter(X_test_simple, y_pred_simple, alpha=0.5, color='red', label='Prédictions')
plt.xlabel(best_single_feature)
plt.ylabel('Salaire (€)')
plt.title('Régression Linéaire Simple')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 7.3 Régression Linéaire Multiple

```python
print("\n=== RÉGRESSION LINÉAIRE MULTIPLE ===")

# Utiliser toutes les features
model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train)

# Prédictions
y_pred_multiple = model_multiple.predict(X_test_scaled)

# Évaluation
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print(f"Nombre de features: {X_train_scaled.shape[1]}")
print(f"RMSE: {rmse_multiple:.0f} € (simple: {rmse_simple:.0f})")
print(f"MAE: {mae_multiple:.0f} €")
print(f"R²: {r2_multiple:.3f} (simple: {r2_simple:.3f})")

# Top coefficients
coef_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': model_multiple.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n=== TOP 10 COEFFICIENTS ===")
print(coef_df.head(10))
```

### 7.4 Régression Polynomiale

```python
print("\n=== RÉGRESSION POLYNOMIALE ===")

# Sélectionner les features numériques continues importantes
poly_features = ['experience', 'age', 'performance_score', 'hours_per_week']
X_train_poly = X_train_scaled[poly_features]
X_test_poly = X_test_scaled[poly_features]

# Créer les features polynomiales (degré 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_features = poly.fit_transform(X_train_poly)
X_test_poly_features = poly.transform(X_test_poly)

print(f"Features originales: {X_train_poly.shape[1]}")
print(f"Features polynomiales: {X_train_poly_features.shape[1]}")

# Entraîner le modèle
model_poly = LinearRegression()
model_poly.fit(X_train_poly_features, y_train)

# Prédictions
y_pred_poly = model_poly.predict(X_test_poly_features)

# Évaluation
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"RMSE: {rmse_poly:.0f} € (multiple: {rmse_multiple:.0f})")
print(f"MAE: {mae_poly:.0f} €")
print(f"R²: {r2_poly:.3f} (multiple: {r2_multiple:.3f})")

# Attention au sur-ajustement
r2_train_poly = model_poly.score(X_train_poly_features, y_train)
print(f"\nR² Train: {r2_train_poly:.3f}")
print(f"R² Test: {r2_poly:.3f}")
if r2_train_poly - r2_poly > 0.1:
    print("⚠️ ATTENTION: Possible sur-ajustement!")
```

---

## 📊 Étape 8: Analyse des Résultats

### 8.1 Comparaison des modèles

```python
# Créer un tableau de comparaison
results = pd.DataFrame({
    'Modèle': ['Baseline (Moyenne)', 'Linéaire Simple', 'Linéaire Multiple', 'Polynomiale'],
    'RMSE (€)': [rmse_baseline, rmse_simple, rmse_multiple, rmse_poly],
    'MAE (€)': [mae_baseline, mae_simple, mae_multiple, mae_poly],
    'R²': [r2_baseline, r2_simple, r2_multiple, r2_poly]
})

print("\n=== COMPARAISON DES MODÈLES ===")
print(results.to_string(index=False))

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RMSE
axes[0].bar(results['Modèle'], results['RMSE (€)'], color=['gray', 'blue', 'green', 'orange'])
axes[0].set_ylabel('RMSE (€)')
axes[0].set_title('Root Mean Squared Error')
axes[0].tick_params(axis='x', rotation=45)

# MAE
axes[1].bar(results['Modèle'], results['MAE (€)'], color=['gray', 'blue', 'green', 'orange'])
axes[1].set_ylabel('MAE (€)')
axes[1].set_title('Mean Absolute Error')
axes[1].tick_params(axis='x', rotation=45)

# R²
axes[2].bar(results['Modèle'], results['R²'], color=['gray', 'blue', 'green', 'orange'])
axes[2].set_ylabel('R²')
axes[2].set_title('Coefficient de Détermination')
axes[2].tick_params(axis='x', rotation=45)
axes[2].set_ylim([min(0, results['R²'].min() - 0.1), 1])

plt.tight_layout()
plt.show()

# Identifier le meilleur modèle
best_model_idx = results['R²'].idxmax()
print(f"\n🏆 MEILLEUR MODÈLE: {results.loc[best_model_idx, 'Modèle']}")
print(f"   R² = {results.loc[best_model_idx, 'R²']:.3f}")
print(f"   RMSE = {results.loc[best_model_idx, 'RMSE (€)']:.0f} €")
```

### 8.2 Analyse des résidus

```python
# Utiliser le meilleur modèle pour l'analyse des résidus
if 'Multiple' in results.loc[best_model_idx, 'Modèle']:
    y_pred_best = y_pred_multiple
    model_name = "Régression Multiple"
elif 'Polynomiale' in results.loc[best_model_idx, 'Modèle']:
    y_pred_best = y_pred_poly
    model_name = "Régression Polynomiale"
else:
    y_pred_best = y_pred_simple
    model_name = "Régression Simple"

# Calculer les résidus
residuals = y_test - y_pred_best

# Visualisation des résidus
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Résidus vs Prédictions
axes[0, 0].scatter(y_pred_best, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Valeurs Prédites')
axes[0, 0].set_ylabel('Résidus')
axes[0, 0].set_title('Résidus vs Prédictions')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution des résidus
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Résidus')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution des Résidus')
axes[0, 1].axvline(x=0, color='red', linestyle='--')

# 3. Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# 4. Résidus vs Valeurs réelles
axes[1, 1].scatter(y_test, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Valeurs Réelles')
axes[1, 1].set_ylabel('Résidus')
axes[1, 1].set_title('Résidus vs Valeurs Réelles')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Analyse des Résidus - {model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Tests statistiques sur les résidus
print("\n=== ANALYSE STATISTIQUE DES RÉSIDUS ===")
print(f"Moyenne des résidus: {residuals.mean():.2f} € (devrait être proche de 0)")
print(f"Médiane des résidus: {residuals.median():.2f} €")
print(f"Écart-type des résidus: {residuals.std():.2f} €")

# Test de normalité
_, p_value = stats.shapiro(residuals)
print(f"Test de Shapiro-Wilk (normalité): p-value = {p_value:.4f}")
if p_value > 0.05:
    print("→ Les résidus suivent une distribution normale ✅")
else:
    print("→ Les résidus ne suivent PAS une distribution normale ⚠️")
```

### 8.3 Prédictions vs Réalité

```python
# Graphique des prédictions vs réalité
plt.figure(figsize=(12, 6))

# Subplot 1: Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Prédiction parfaite')
plt.xlabel('Salaire Réel (€)')
plt.ylabel('Salaire Prédit (€)')
plt.title(f'Prédictions vs Réalité - {model_name}')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Erreur relative
plt.subplot(1, 2, 2)
error_pct = ((y_pred_best - y_test) / y_test * 100).values
plt.hist(error_pct, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Erreur Relative (%)')
plt.ylabel('Fréquence')
plt.title('Distribution de l\'Erreur Relative')
plt.axvline(x=0, color='red', linestyle='--', label='Erreur nulle')
plt.axvline(x=np.median(error_pct), color='green', linestyle='--',
            label=f'Médiane: {np.median(error_pct):.1f}%')
plt.legend()

plt.tight_layout()
plt.show()

# Statistiques sur les erreurs
print("\n=== STATISTIQUES DES ERREURS ===")
print(f"Erreur absolue moyenne: {mae_multiple:.0f} €")
print(f"Erreur absolue médiane: {np.median(np.abs(residuals)):.0f} €")
print(f"Erreur relative moyenne: {np.mean(np.abs(error_pct)):.1f}%")
print(f"% de prédictions à ±10%: {np.sum(np.abs(error_pct) <= 10) / len(error_pct) * 100:.1f}%")
print(f"% de prédictions à ±20%: {np.sum(np.abs(error_pct) <= 20) / len(error_pct) * 100:.1f}%")
```

---

## 🎯 Étape 9: Interprétation et Insights

### 9.1 Facteurs les plus importants

```python
# Récupérer les coefficients du meilleur modèle (supposons que c'est le multiple)
feature_importance = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': model_multiple.coef_,
    'Abs_Coefficient': np.abs(model_multiple.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("=== TOP 10 FACTEURS INFLUENÇANT LE SALAIRE ===")
for idx, row in feature_importance.head(10).iterrows():
    impact = "augmente" if row['Coefficient'] > 0 else "diminue"
    print(f"{row['Feature']}: {impact} le salaire de {abs(row['Coefficient']):.0f} € par unité")

# Visualisation
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Coefficient (Impact sur le salaire en €)')
plt.title('Impact des Variables sur le Salaire')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

### 9.2 Exemples de prédictions

```python
# Prendre quelques exemples du test set
sample_indices = np.random.choice(y_test.index, 5, replace=False)

print("\n=== EXEMPLES DE PRÉDICTIONS ===")
for idx in sample_indices:
    real_salary = y_test.loc[idx]
    # Trouver l'index dans les arrays de prédiction
    test_position = y_test.index.get_loc(idx)
    predicted_salary = y_pred_best[test_position]
    error = predicted_salary - real_salary
    error_pct = error / real_salary * 100

    print(f"\nEmployé #{idx}:")
    print(f"  Salaire réel: {real_salary:.0f} €")
    print(f"  Salaire prédit: {predicted_salary:.0f} €")
    print(f"  Erreur: {error:+.0f} € ({error_pct:+.1f}%)")

    # Afficher quelques caractéristiques importantes
    original_data = df_clean.loc[idx]
    print(f"  Profil: {original_data['department']}, {original_data['education']}")
    print(f"  Experience: {original_data['experience']:.0f} ans")
    print(f"  Performance: {original_data['performance_score']:.1f}/10")
```

---

## 💡 Étape 10: Améliorations Possibles

### 10.1 Suggestions d'amélioration

```python
print("=== SUGGESTIONS D'AMÉLIORATION ===")

print("\n1. FEATURE ENGINEERING AVANCÉ:")
print("   - Interactions entre variables (ex: experience × education)")
print("   - Transformations logarithmiques pour les variables skewed")
print("   - Binning intelligent des variables continues")

print("\n2. SÉLECTION DE FEATURES:")
print("   - Utiliser SelectKBest ou RFE pour sélectionner les meilleures features")
print("   - Éliminer les features avec faible importance")

print("\n3. RÉGULARISATION:")
print("   - Essayer Ridge ou Lasso pour éviter le sur-ajustement")
print("   - Cross-validation pour optimiser les hyperparamètres")

print("\n4. MODÈLES AVANCÉS:")
print("   - Random Forest Regressor")
print("   - Gradient Boosting (XGBoost, LightGBM)")
print("   - Réseaux de neurones")

print("\n5. VALIDATION:")
print("   - K-fold cross-validation")
print("   - Validation temporelle si les données ont une composante temporelle")
```

### 10.2 Code pour aller plus loin

```python
# Exemple: Régression Ridge avec cross-validation
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("\n=== TEST RAPIDE: RÉGRESSION RIDGE ===")

# Tester différentes valeurs d'alpha
alphas = [0.1, 1, 10, 100, 1000]
best_alpha = None
best_score = -np.inf

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5,
                             scoring='r2', n_jobs=-1)
    mean_score = scores.mean()
    print(f"Alpha={alpha}: R² CV moyen = {mean_score:.3f}")

    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"\nMeilleur alpha: {best_alpha}")

# Entraîner le modèle final
ridge_final = Ridge(alpha=best_alpha)
ridge_final.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_final.predict(X_test_scaled)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"R² Ridge sur test: {r2_ridge:.3f}")
```

---

## ✅ Étape 11: Conclusion et Export

### 11.1 Résumé des résultats

```python
print("\n" + "="*60)
print("RÉSUMÉ FINAL")
print("="*60)

print(f"""
📊 Dataset:
   - {len(df)} employés analysés
   - {X.shape[1]} features utilisées après preprocessing
   - Variable cible: Salaire (€)

🔧 Prétraitement effectué:
   ✅ Valeurs manquantes imputées
   ✅ Formats harmonisés
   ✅ Variables catégorielles encodées
   ✅ Outliers traités
   ✅ Features normalisées

📈 Modèles testés:
   1. Baseline (moyenne): R² = {r2_baseline:.3f}
   2. Régression Simple: R² = {r2_simple:.3f}
   3. Régression Multiple: R² = {r2_multiple:.3f}
   4. Régression Polynomiale: R² = {r2_poly:.3f}

🏆 Meilleur modèle: {results.loc[best_model_idx, 'Modèle']}
   - R² = {results.loc[best_model_idx, 'R²']:.3f}
   - RMSE = {results.loc[best_model_idx, 'RMSE (€)']:.0f} €
   - MAE = {results.loc[best_model_idx, 'MAE (€)']:.0f} €

💡 Principaux facteurs influençant le salaire:
""")

for idx, row in feature_importance.head(5).iterrows():
    print(f"   - {row['Feature']}: {row['Coefficient']:+.0f} € par unité")
```

### 11.2 Sauvegarder le modèle

```python
import pickle

# Sauvegarder le meilleur modèle
with open('model_salaire.pkl', 'wb') as f:
    pickle.dump(model_multiple, f)

# Sauvegarder le scaler
with open('scaler_salaire.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Modèle et scaler sauvegardés!")

# Comment réutiliser le modèle
print("\n=== POUR RÉUTILISER LE MODÈLE ===")
print("""
# Charger le modèle et le scaler
with open('model_salaire.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler_salaire.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Préparer les nouvelles données (même format que X)
new_data = ...  # DataFrame avec les mêmes colonnes
new_data_scaled = scaler.transform(new_data)

# Faire des prédictions
predictions = model.predict(new_data_scaled)
""")
```

---

## 🎯 Exercices Supplémentaires

### Exercice 1: Analyse approfondie
1. Créez des graphiques pour chaque département montrant la relation salaire/experience
2. Identifiez les employés sous-payés et sur-payés (résidus extrêmes)
3. Analysez si le modèle est biaisé pour certains groupes

### Exercice 2: Feature Engineering avancé
1. Créez des termes d'interaction (ex: experience × education_encoded)
2. Appliquez des transformations (log, sqrt) aux variables skewed
3. Créez des features basées sur des ratios

### Exercice 3: Modèles avancés
1. Implémentez une régression Lasso et comparez avec Ridge
2. Utilisez un Random Forest et analysez l'importance des features
3. Créez un ensemble model combinant plusieurs approches

### Exercice 4: Validation robuste
1. Implémentez une validation croisée k-fold
2. Créez des courbes d'apprentissage
3. Analysez la stabilité du modèle avec bootstrap

---

## 🚨 Points Clés à Retenir

1. **Preprocessing est crucial**: 80% du travail, 80% de l'impact!
2. **Commencer simple**: Baseline → Simple → Multiple → Complexe
3. **Validation rigoureuse**: Train/Test split AVANT tout preprocessing statistique
4. **Métriques multiples**: R², RMSE, MAE donnent différentes perspectives
5. **Analyse des résidus**: Révèle les problèmes du modèle
6. **Interprétabilité**: Un modèle simple et interprétable vaut souvent mieux qu'un modèle complexe
7. **Itération**: Le machine learning est un processus itératif

---

## 📝 Notes Finales

Ce TD vous a guidé à travers tout le pipeline d'un projet de régression réaliste:
- Données imparfaites nécessitant du nettoyage
- Multiple approches de modélisation
- Évaluation rigoureuse
- Interprétation business

Les compétences acquises ici sont directement applicables en entreprise!

**Félicitations! Vous avez complété le TD de régression! 🎉**

Pour aller plus loin:
- Essayez d'autres datasets
- Explorez des modèles plus avancés (XGBoost, Neural Networks)
- Participez à des compétitions Kaggle

**Bon courage pour la suite de votre parcours en Data Science! 🚀**