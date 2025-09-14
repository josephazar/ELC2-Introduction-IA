# 🚢 TP: Prétraitement des Données - Dataset Titanic

## 📋 Objectif
Apprendre à préparer des données réelles pour l'analyse en utilisant le célèbre dataset du Titanic. Vous allez nettoyer, transformer et préparer ces données étape par étape.

## 🎯 Ce que vous allez apprendre
- Comment charger et explorer des données
- Identifier et traiter les valeurs manquantes
- Encoder les variables catégorielles
- Détecter et gérer les outliers
- Normaliser les features
- Éviter le data leakage avec train/test split

---

## 📚 Étape 0: Préparation de l'environnement

### Importation des bibliothèques nécessaires

```python
# Copier ce code au début de votre fichier Python
import pandas as pd          # Pour manipuler les données
import numpy as np           # Pour les calculs mathématiques
import matplotlib.pyplot as plt  # Pour les graphiques
import seaborn as sns        # Pour de beaux graphiques

# Configuration pour de meilleurs affichages
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.max_rows', 100)      # Afficher plus de lignes
```

**💡 Pourquoi ces bibliothèques?**
- `pandas`: C'est Excel en Python! Pour lire, manipuler des tableaux de données
- `numpy`: Pour les calculs mathématiques (moyenne, médiane, etc.)
- `matplotlib/seaborn`: Pour visualiser vos données

---

## 📊 Étape 1: Charger et Explorer les Données

### 1.1 Chargement du dataset

```python
# Charger le fichier CSV
df = pd.read_csv('data/titanic.csv')

# Afficher les premières lignes
print("=== PREMIÈRES LIGNES DU DATASET ===")
print(df.head())
```

### 1.2 Comprendre la structure des données

```python
# Informations générales sur le dataset
print("\n=== INFORMATIONS GÉNÉRALES ===")
print(df.info())

# Dimensions du dataset
print(f"\nNombre de passagers: {df.shape[0]}")
print(f"Nombre de variables: {df.shape[1]}")

# Liste des colonnes
print("\n=== COLONNES DISPONIBLES ===")
print(df.columns.tolist())
```

### 1.3 Comprendre chaque variable

```python
# Description statistique des variables numériques
print("\n=== STATISTIQUES DES VARIABLES NUMÉRIQUES ===")
print(df.describe())

# Types de données
print("\n=== TYPES DE DONNÉES ===")
print(df.dtypes)
```

### 📝 Questions à se poser:
1. Combien y a-t-il de passagers?
2. Quelles sont les variables numériques? Catégorielles?
3. Y a-t-il des valeurs manquantes?

---

## 🔍 Étape 2: Analyse des Valeurs Manquantes

### 2.1 Identifier les valeurs manquantes

```python
# Compter les valeurs manquantes par colonne
print("\n=== VALEURS MANQUANTES PAR COLONNE ===")
missing_values = df.isnull().sum()
print(missing_values)

# Pourcentage de valeurs manquantes
print("\n=== POURCENTAGE DE VALEURS MANQUANTES ===")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))

# Visualisation des valeurs manquantes
plt.figure(figsize=(10, 6))
missing_percentage.plot(kind='bar')
plt.title("Pourcentage de valeurs manquantes par colonne")
plt.ylabel("Pourcentage (%)")
plt.xticks(rotation=45)
plt.show()
```

### 2.2 Décider de la stratégie pour chaque colonne

**🤔 Réflexion pour chaque colonne avec des valeurs manquantes:**

#### age (177 valeurs manquantes - ~20%)
```python
# Stratégie: Imputer avec la médiane (robuste aux outliers)
print("\n=== TRAITEMENT DE L'ÂGE ===")
print(f"Médiane de l'âge: {df['age'].median()}")
print(f"Moyenne de l'âge: {df['age'].mean():.2f}")

# Imputation avec la médiane
df['age'].fillna(df['age'].median(), inplace=True)
print(f"Valeurs manquantes après imputation: {df['age'].isnull().sum()}")
```

#### deck (688 valeurs manquantes - ~77%)
```python
# Stratégie: Créer une variable indicatrice et remplir avec 'Unknown'
print("\n=== TRAITEMENT DE DECK ===")
# Créer une colonne pour indiquer si le passager avait un deck connu
df['Has_Deck'] = df['deck'].notna().astype(int)
print(f"Passagers avec deck connu: {df['Has_Deck'].sum()}")
print(f"Passagers sans deck: {(df['Has_Deck'] == 0).sum()}")

# Remplir les valeurs manquantes avec 'Unknown'
df['deck'].fillna('Unknown', inplace=True)
print("Valeurs manquantes de 'deck' remplacées par 'Unknown'")
```

#### embarked et embark_town (2 valeurs manquantes)
```python
# Stratégie: Imputer avec le mode (valeur la plus fréquente)
print("\n=== TRAITEMENT DE EMBARKED ===")
mode_embarked = df['embarked'].mode()[0]
print(f"Port d'embarquement le plus fréquent: {mode_embarked}")

df['embarked'].fillna(mode_embarked, inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
print(f"Valeurs manquantes après imputation: {df['embarked'].isnull().sum()}")
```

### 2.3 Vérification finale

```python
# Vérifier qu'il n'y a plus de valeurs manquantes
print("\n=== VÉRIFICATION FINALE DES VALEURS MANQUANTES ===")
print(df.isnull().sum())
```

---

## 🏷️ Étape 3: Encodage des Variables Catégorielles

### 3.1 Identifier les variables catégorielles

```python
# Identifier les colonnes catégorielles (type 'object')
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\n=== VARIABLES CATÉGORIELLES ===")
print(categorical_columns)

# Pour chaque variable catégorielle, voir les valeurs uniques
for col in categorical_columns:
    print(f"\n{col}: {df[col].unique()}")
```

### 3.2 Encodage selon le type de variable

#### Variables binaires (2 catégories) → Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# sex: male/female → 0/1
print("\n=== ENCODAGE DE 'SEX' ===")
le_sex = LabelEncoder()
df['sex_encoded'] = le_sex.fit_transform(df['sex'])
print(f"Mapping: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# Supprimer l'ancienne colonne
df.drop('sex', axis=1, inplace=True)
```

#### Variables nominales (>2 catégories sans ordre) → One-Hot Encoding

```python
# embarked: S/C/Q → One-Hot
print("\n=== ONE-HOT ENCODING DE 'EMBARKED' ===")
df_embarked = pd.get_dummies(df['embarked'], prefix='embarked')
print(df_embarked.head())

# Ajouter les nouvelles colonnes et supprimer l'ancienne
df = pd.concat([df, df_embarked], axis=1)
df.drop('embarked', axis=1, inplace=True)
print("One-Hot encoding complété pour embarked")
```

### 3.3 Supprimer les colonnes non nécessaires

```python
# Colonnes à supprimer (colonnes redondantes ou non utiles)
columns_to_drop = ['Unnamed: 0', 'embark_town', 'alive', 'who', 'adult_male', 'class']
print(f"\n=== SUPPRESSION DES COLONNES NON NÉCESSAIRES ===")
print(f"Colonnes à supprimer: {columns_to_drop}")

df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
print(f"Nouvelles dimensions: {df.shape}")
```

---

## 📈 Étape 4: Détection et Traitement des Outliers

### 4.1 Visualiser les distributions

```python
# Visualiser les variables numériques
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n=== VISUALISATION DES DISTRIBUTIONS ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:6]):
    axes[idx].boxplot(df[col].dropna())
    axes[idx].set_title(f'Boxplot de {col}')
    axes[idx].set_ylabel('Valeur')

plt.tight_layout()
plt.show()
```

### 4.2 Détecter les outliers avec la méthode IQR

```python
def detect_outliers_iqr(data, column):
    """
    Détecte les outliers en utilisant la méthode IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    print(f"\n=== OUTLIERS POUR {column} ===")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Bornes: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Nombre d'outliers: {len(outliers)}")

    return outliers, lower_bound, upper_bound

# Analyser les outliers pour 'fare'
outliers_fare, lower, upper = detect_outliers_iqr(df, 'fare')
```

### 4.3 Traiter les outliers (Capping)

```python
# Capping: remplacer les outliers par les bornes
def cap_outliers(data, column, lower_bound, upper_bound):
    """
    Remplace les outliers par les bornes min/max
    """
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

# Appliquer le capping sur fare
print(f"\nValeur max avant capping: {df['fare'].max():.2f}")
df = cap_outliers(df, 'fare', lower, upper)
print(f"Valeur max après capping: {df['fare'].max():.2f}")
```

---

## ⚖️ Étape 5: Feature Engineering (Création de Variables)

### 5.1 Créer des variables pertinentes

```python
# Taille de la famille
print("\n=== CRÉATION DE NOUVELLES FEATURES ===")
df['family_size'] = df['sibsp'] + df['parch'] + 1
print(f"Taille de famille moyenne: {df['family_size'].mean():.2f}")

# Note: 'alone' existe déjà dans le dataset
print(f"Passagers seuls: {df['alone'].sum()}")

# Catégorie d'âge
df['age_category'] = pd.cut(df['age'],
                           bins=[0, 12, 18, 60, 100],
                           labels=['Enfant', 'Adolescent', 'Adulte', 'Senior'])

# Encoder la catégorie d'âge
df = pd.get_dummies(df, columns=['age_category'], prefix='age')
print("Nouvelles features créées!")
```

---

## 🔄 Étape 6: Division Train/Test (CRUCIAL!)

### 6.1 Séparer features et target

```python
# 'survived' est notre variable cible
if 'survived' in df.columns:
    # Séparer X (features) et y (target)
    X = df.drop('survived', axis=1)
    y = df['survived']
else:
    # Si pas de target, on prend toutes les colonnes
    X = df
    y = None

print(f"\n=== DIMENSIONS DES DONNÉES ===")
print(f"X shape: {X.shape}")
if y is not None:
    print(f"y shape: {y.shape}")
```

### 6.2 Division train/test

```python
from sklearn.model_selection import train_test_split

if y is not None:
    # Division 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n=== DIVISION TRAIN/TEST ===")
    print(f"Train set: {X_train.shape[0]} échantillons")
    print(f"Test set: {X_test.shape[0]} échantillons")
else:
    # Si pas de target, on garde tout en train
    X_train = X
    X_test = None
```

---

## 📏 Étape 7: Normalisation des Features

### 7.1 Pourquoi normaliser?

```python
# Voir les échelles différentes
print("\n=== ÉCHELLES DES VARIABLES ===")
print(X_train.describe().loc[['min', 'max']])
```

### 7.2 Appliquer la normalisation

```python
from sklearn.preprocessing import StandardScaler

# Créer le scaler
scaler = StandardScaler()

# IMPORTANT: fit sur train, transform sur train ET test
X_train_scaled = scaler.fit_transform(X_train)

if X_test is not None:
    X_test_scaled = scaler.transform(X_test)  # PAS fit_transform!

print("\n=== APRÈS NORMALISATION ===")
print(f"Moyenne du train (devrait être ~0): {X_train_scaled.mean():.4f}")
print(f"Écart-type du train (devrait être ~1): {X_train_scaled.std():.4f}")
```

---

## ✅ Étape 8: Vérification Finale

### 8.1 Résumé du preprocessing

```python
print("\n" + "="*50)
print("RÉSUMÉ DU PRÉTRAITEMENT")
print("="*50)

print(f"""
Étapes complétées:
1. ✅ Données chargées: {df.shape[0]} passagers
2. ✅ Valeurs manquantes traitées
3. ✅ Variables catégorielles encodées
4. ✅ Outliers détectés et traités
5. ✅ Nouvelles features créées
6. ✅ Division train/test effectuée
7. ✅ Normalisation appliquée

Dimensions finales:
- Train: {X_train_scaled.shape if 'X_train_scaled' in locals() else 'N/A'}
- Test: {X_test_scaled.shape if 'X_test_scaled' in locals() else 'N/A'}
""")
```

### 8.2 Sauvegarder les données prétraitées

```python
# Convertir en DataFrame pour sauvegarder
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
if X_test is not None:
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarder en CSV
X_train_final.to_csv('titanic_train_processed.csv', index=False)
if X_test is not None:
    X_test_final.to_csv('titanic_test_processed.csv', index=False)

print("\n✅ Données prétraitées sauvegardées!")
```

---

## 🎯 Exercices Supplémentaires

### Exercice 1: Analyse approfondie
1. Calculez manuellement la médiane de l'âge (sans pandas)
2. Identifiez le passager avec le billet le plus cher
3. Quelle classe avait le plus de passagers?

### Exercice 2: Feature Engineering avancé
1. Créez une variable "Title" à partir du nom (Mr., Mrs., etc.)
2. Créez une variable "Deck" à partir de Cabin (avant de la supprimer)
3. Créez des interactions entre variables (ex: Age × Sex)

### Exercice 3: Visualisations
1. Créez un heatmap de corrélation
2. Visualisez la distribution de l'âge par classe
3. Comparez les tarifs par port d'embarquement

---

## 🚨 Erreurs Communes à Éviter

1. **Data Leakage**: Ne JAMAIS calculer des statistiques sur le test set
2. **Oublier de sauvegarder le scaler**: Vous en aurez besoin pour de nouvelles données
3. **Supprimer trop de données**: Parfois mieux d'imputer que de supprimer
4. **One-Hot sur trop de catégories**: Peut créer trop de colonnes
5. **Normaliser avant split**: Toujours split d'abord!

---

## 📝 Notes Finales

- Ce preprocessing est la base pour tout projet de Data Science
- 80% du travail en ML est du preprocessing!
- Documentez toujours vos choix (pourquoi médiane vs moyenne?)
- Testez différentes approches et comparez

**Bravo! Vous avez complété le preprocessing du Titanic! 🎉**