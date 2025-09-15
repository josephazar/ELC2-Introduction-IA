# ============================================================================
# PRÉTRAITEMENT DES DONNÉES - GUIDE COMPLET POUR DÉBUTANTS
# ============================================================================
# Ce fichier montre étape par étape comment préparer des données pour
# l'apprentissage automatique. Chaque transformation est expliquée en détail.
# ============================================================================

# Importation des bibliothèques nécessaires
import numpy as np  # Pour les calculs mathématiques et les tableaux
import matplotlib.pyplot as plt  # Pour créer des graphiques (non utilisé ici)
import pandas as pd  # Pour manipuler les données sous forme de tableaux
from scipy import stats  # Pour les calculs statistiques (Z-score)

"""## Importation du dataset"""

# Chargement du fichier CSV contenant nos données
# pd.read_csv() lit le fichier et le transforme en DataFrame (tableau structuré)
dataset = pd.read_csv('data/Data.csv')

# Affichage des premières lignes pour comprendre la structure
print("Dataset Original:")
print(dataset.head(10))  # .head(10) affiche les 10 premières lignes

# Informations sur la taille du dataset
print(f"\nDimensions du dataset: {dataset.shape}")  # (nombre de lignes, nombre de colonnes)

# Informations détaillées sur chaque colonne
print(f"\nInformations sur le dataset:")
print(dataset.info())  # Type de données, valeurs non-nulles, etc.

"""## Analyse des données manquantes"""

print("\n" + "="*50)
print("ANALYSE DES DONNÉES MANQUANTES")
print("="*50)

# .isnull() détecte les valeurs manquantes (True si manquant, False sinon)
# .sum() compte le nombre de True pour chaque colonne
print("\nNombre de valeurs manquantes par colonne:")
print(dataset.isnull().sum())

# Calcul du pourcentage de données manquantes
# Important pour décider si on supprime ou impute les valeurs
print(f"\nPourcentage de valeurs manquantes:")
print((dataset.isnull().sum() / len(dataset)) * 100)

"""## Détection des valeurs aberrantes (outliers) avec Z-Score"""

print("\n" + "="*50)
print("DÉTECTION DES OUTLIERS (Méthode Z-Score)")
print("="*50)

# Le Z-score mesure à combien d'écarts-types une valeur se trouve de la moyenne
# Z = (valeur - moyenne) / écart-type
# Si |Z| > 3, la valeur est considérée comme aberrante

# .dropna() enlève temporairement les valeurs manquantes pour le calcul
z_scores = np.abs(stats.zscore(dataset['Salary'].dropna()))

# Seuil standard: 3 écarts-types
threshold = 3

# np.where trouve les indices où la condition est vraie
outlier_indices = np.where(z_scores > threshold)[0]

print(f"\nValeurs aberrantes détectées dans la colonne Salary (Z-score > {threshold}):")
salary_outliers = dataset['Salary'].dropna().iloc[outlier_indices]

# Affichage des outliers avec leur position dans le dataset
for idx, value in zip(dataset[dataset['Salary'].isin(salary_outliers)].index, salary_outliers):
    print(f"  Ligne {idx}: ${value:,.0f}")

"""## Analyse des corrélations entre variables"""

print("\n" + "="*50)
print("ANALYSE DE CORRÉLATION")
print("="*50)

# Les variables fortement corrélées apportent la même information
# On peut en supprimer une pour simplifier le modèle

# Sélection uniquement des colonnes numériques
numeric_cols = dataset.select_dtypes(include=[np.number]).columns
correlation_matrix = dataset[numeric_cols].corr()

# Calcul de la corrélation entre Age et Experience
print("\nCorrélation entre Age et Experience:")
age_exp_corr = dataset[['Age', 'Experience']].corr().iloc[0, 1]
print(f"  Coefficient de corrélation: {age_exp_corr:.3f}")
print(f"  Interprétation: Corrélation {'négative' if age_exp_corr < 0 else 'positive'} forte")
print(f"  Note: Experience ≈ Age - années d'études - 18, d'où la forte relation")

"""## Préparation des features (X) et de la cible (y)"""

# X contient toutes les colonnes SAUF la dernière (features/caractéristiques)
# iloc[:, :-1] = toutes les lignes (:), toutes les colonnes sauf la dernière (:-1)
X = dataset.iloc[:, :-1].values

# y contient uniquement la dernière colonne (target/cible à prédire)
# iloc[:, -1] = toutes les lignes (:), dernière colonne (-1)
y = dataset.iloc[:, -1].values

print("\n" + "="*50)
print("ÉTAPES DE PRÉTRAITEMENT")
print("="*50)

print("\nX original (5 premières lignes):")
print(X[:5])

"""## 1. Traitement des valeurs manquantes - Imputation"""

print("\n1. TRAITEMENT DES VALEURS MANQUANTES")
print("-" * 30)

from sklearn.impute import SimpleImputer

# IMPUTATION DE L'ÂGE
# La médiane est plus robuste aux outliers que la moyenne
# Exemple: [20, 25, 30, 100] → moyenne=43.75, médiane=27.5
print("\nImputation de l'Age avec la médiane (robuste aux outliers):")
age_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# X[:, 1:2] sélectionne la colonne Age (index 1)
# On garde :2 pour avoir une matrice 2D (nécessaire pour sklearn)
X[:, 1:2] = age_imputer.fit_transform(X[:, 1:2])

# IMPUTATION DU SALAIRE
print("Imputation du Salary avec la moyenne:")
salary_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 2:3] = salary_imputer.fit_transform(X[:, 2:3])

# Vérification que l'imputation a fonctionné
print("\nX après imputation (lignes qui avaient des valeurs manquantes):")
missing_age_rows = [7, 12, 15, 18, 22, 25, 29, 32, 36, 39, 43, 46, 50]
for i in missing_age_rows[:3]:  # Afficher seulement 3 exemples
    if i < len(X):
        print(f"  Ligne {i}: Age={X[i, 1]:.1f}, Salary={X[i, 2]:.0f}")

"""## 2. Traitement des outliers - Méthode du Capping"""

print("\n2. TRAITEMENT DES OUTLIERS")
print("-" * 30)

# Conversion en float pour les calculs
salaries = X[:, 2].astype(float)

# MÉTHODE IQR (Interquartile Range)
# Q1 = 25e percentile, Q3 = 75e percentile
Q1 = np.percentile(salaries, 25)
Q3 = np.percentile(salaries, 75)
IQR = Q3 - Q1  # Écart interquartile

# Règle standard: outlier si valeur < Q1-1.5*IQR ou > Q3+1.5*IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nStatistiques des salaires:")
print(f"  Q1 (25e percentile): ${Q1:,.0f}")
print(f"  Q3 (75e percentile): ${Q3:,.0f}")
print(f"  IQR (écart interquartile): ${IQR:,.0f}")
print(f"  Borne inférieure: ${lower_bound:,.0f}")
print(f"  Borne supérieure: ${upper_bound:,.0f}")

# Comptage des outliers
outlier_count = np.sum((salaries < lower_bound) | (salaries > upper_bound))
print(f"\nNombre d'outliers trouvés: {outlier_count}")

# CAPPING: on remplace les outliers par la borne supérieure
print("\nCapping des outliers à la borne supérieure:")
original_outliers = salaries[salaries > upper_bound].copy()
salaries[salaries > upper_bound] = upper_bound
X[:, 2] = salaries

# Affichage des transformations
for i, orig in enumerate(original_outliers[:3]):  # 3 exemples
    print(f"  ${orig:,.0f} → ${upper_bound:,.0f}")

"""## 3. Encodage des variables catégorielles"""

print("\n3. ENCODAGE DES VARIABLES CATÉGORIELLES")
print("-" * 30)

"""### 3a. One-Hot Encoding pour Country (variable nominale)"""

# Variables nominales = pas d'ordre naturel (ex: pays, couleurs)
# On crée une colonne binaire pour chaque catégorie

print("\n3a. One-Hot Encoding pour Country (catégorielle nominale):")
print("    Pays: France, Spain, Germany, USA, Canada, UK")
print("    Chaque pays obtient sa propre colonne binaire (0 ou 1)")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ColumnTransformer applique différentes transformations à différentes colonnes
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],  # OneHot sur colonne 0 (Country)
    remainder='passthrough'  # Garder les autres colonnes telles quelles
)
X = np.array(ct.fit_transform(X))

print(f"\n    Dimensions après One-Hot Encoding: {X.shape}")
print("    Première ligne (exemple France):", X[0, :10])

"""### 3b. Label Encoding pour Education (variable ordinale)"""

# Variables ordinales = avec ordre naturel (ex: taille S<M<L, niveau d'études)
# On assigne un nombre entier à chaque catégorie selon l'ordre

print("\n3b. Label Encoding pour Education (catégorielle ordinale):")
print("    Les niveaux d'éducation ont un ordre naturel:")
print("    HighSchool < Bachelor < Master < PhD")

from sklearn.preprocessing import LabelEncoder

# Mapping manuel pour comprendre l'encodage
education_mapping = {'HighSchool': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
education_col_idx = 8  # Après one-hot encoding, Education est à l'index 8

# Récupération des valeurs d'éducation
education_values = X[:, education_col_idx]

# Création et application du LabelEncoder
le_education = LabelEncoder()
le_education.fit(['HighSchool', 'Bachelor', 'Master', 'PhD'])
X[:, education_col_idx] = le_education.transform(education_values)

print("\n    Mapping de l'encodage:")
for edu, code in education_mapping.items():
    print(f"      {edu} → {code}")

"""### 3c. Encodage de la variable cible (target)"""

print("\n3c. Label Encoding pour Purchased (variable cible):")
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print("    No → 0, Yes → 1")

"""## 4. Analyse et suppression des features corrélées"""

print("\n4. CORRÉLATION ET SÉLECTION DE FEATURES")
print("-" * 30)

# Indices des colonnes Age et Experience après transformations
age_idx = 6  # Après one-hot encoding, Age est à l'index 6
experience_idx = 9  # Après one-hot encoding, Experience est à l'index 9

# Calcul du coefficient de corrélation de Pearson
ages = X[:, age_idx].astype(float)
experiences = X[:, experience_idx].astype(float)
correlation = np.corrcoef(ages, experiences)[0, 1]

print(f"\nCorrélation entre Age et Experience: {correlation:.3f}")
print("Puisque ces features sont fortement corrélées, on peut en supprimer une")
print("pour éviter la redondance et simplifier le modèle")
print("Suppression de Experience pour la démonstration...")

# Suppression de la colonne Experience
X = np.delete(X, experience_idx, axis=1)
print(f"\nDimensions après suppression de Experience: {X.shape}")

"""## 5. Division en ensemble d'entraînement et de test"""

print("\n5. DIVISION TRAIN-TEST")
print("-" * 30)

from sklearn.model_selection import train_test_split

# Division 80% entraînement, 20% test
# random_state=42 assure la reproductibilité (même division à chaque exécution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTaille de l'ensemble d'entraînement: {X_train.shape[0]} échantillons")
print(f"Taille de l'ensemble de test: {X_test.shape[0]} échantillons")
print(f"Nombre de features par échantillon: {X_train.shape[1]}")

"""## 6. Normalisation des features (Feature Scaling)"""

print("\n6. NORMALISATION DES FEATURES")
print("-" * 30)

# POURQUOI NORMALISER?
# Les algorithmes basés sur les distances (KNN, SVM, réseaux de neurones)
# sont sensibles aux échelles différentes des features
# Ex: Age (20-50) vs Salaire (40k-100k) → le salaire dominerait

print("\nApplication du StandardScaler (normalisation Z-score):")
print("  - Centre les données autour de moyenne=0")
print("  - Échelonne à écart-type=1")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# IMPORTANT: fit sur train, transform sur train ET test
# fit_transform sur train: calcule moyenne/écart-type ET transforme
X_train = sc.fit_transform(X_train)

# transform seulement sur test: utilise moyenne/écart-type du train
# Évite la fuite de données (data leakage)
X_test = sc.transform(X_test)

print("\nValeurs normalisées (première ligne du train):")
print(f"  5 premières features: {X_train[0, :5]}")

# Alternative: MinMaxScaler
print("\nAlternative: MinMaxScaler pour une plage [0,1]")
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_minmax = mms.fit_transform(X_train)
print(f"  Exemple MinMax: {X_train_minmax[0, :5]}")

"""## Résumé Final"""

print("\n" + "="*50)
print("RÉSUMÉ DU PRÉTRAITEMENT")
print("="*50)

print(f"""
Techniques de prétraitement appliquées:

1. DONNÉES MANQUANTES:
   - Age imputé avec la médiane (robuste aux outliers)
   - Salaire imputé avec la moyenne

2. TRAITEMENT DES OUTLIERS:
   - Détection par Z-score et méthode IQR
   - Capping des salaires extrêmes à la borne supérieure

3. ENCODAGE CATÉGORIEL:
   - One-Hot Encoding pour Country (nominal)
   - Label Encoding pour Education (ordinal)
   - Label Encoding pour la variable cible

4. INGÉNIERIE DES FEATURES:
   - Corrélation élevée identifiée entre Age et Experience
   - Suppression de la feature redondante Experience

5. DIVISION DES DONNÉES:
   - 80% entraînement, 20% test

6. NORMALISATION:
   - StandardScaler pour distribution normale
   - Alternative MinMaxScaler pour plage bornée

Dimensions finales des données prétraitées: {X_train.shape}
""")

print("\n🎯 Les données sont maintenant prêtes pour l'apprentissage automatique!")
print("   Prochaine étape: Entraîner un modèle de classification ou régression")