# ============================================================================
# EXERCICE 1.02 - TRAITEMENT DES VALEURS MANQUANTES ET OUTLIERS
# ============================================================================
# Cet exercice montre comment:
# 1. Créer et traiter des valeurs manquantes
# 2. Détecter et gérer les outliers
# 3. Encoder les variables catégorielles
# 4. Normaliser et standardiser les données
# ============================================================================

# Importation des bibliothèques
import seaborn as sns  # Pour charger des datasets d'exemple
import numpy as np     # Pour les calculs mathématiques
import pandas as pd    # Pour la manipulation de données

# ============================================================================
# PARTIE 1: CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

# Chargement du dataset 'tips'
tips = pd.read_csv("data/tips.csv")

# ============================================================================
# PARTIE 2: CRÉATION ET TRAITEMENT DE VALEURS MANQUANTES
# ============================================================================

print("="*60)
print("TRAITEMENT DES VALEURS MANQUANTES")
print("="*60)

# Extraction de la colonne 'size' (nombre de personnes à table)
size = tips["size"]

# SIMULATION: On crée artificiellement des valeurs manquantes
# .loc[:15] sélectionne les lignes 0 à 15
# np.nan représente une valeur manquante (Not a Number)
size.loc[:15] = np.nan

# Affichage des 20 premières valeurs pour voir les NaN
print("\nColonne 'size' avec valeurs manquantes créées:")
print(size.head(20))
print()

# Vérification de la forme du vecteur
print(f"Dimensions: {size.shape}")
print()

# Comptage des valeurs manquantes
# .isnull() retourne True pour chaque NaN
# .sum() compte le nombre de True
missing_count = size.isnull().sum()
print(f"Nombre de valeurs manquantes: {missing_count}")
print(f"Pourcentage de données manquantes: {(missing_count/len(size))*100:.1f}%")
print()

# ============================================================================
# IMPUTATION DES VALEURS MANQUANTES
# ============================================================================

# Calcul de la moyenne (en ignorant les NaN)
mean = size.mean()  # pandas ignore automatiquement les NaN
mean = round(mean)  # Arrondi car 'size' doit être un entier
print(f"Moyenne des valeurs non-manquantes: {mean}")
print()

# Remplacement des NaN par la moyenne
# inplace=True modifie directement la variable sans créer de copie
size.fillna(mean, inplace=True)

# Vérification après imputation
print("Colonne 'size' après imputation:")
print(size.head(20))
print(f"Valeurs manquantes restantes: {size.isnull().sum()}")
print()

# ============================================================================
# PARTIE 3: DÉTECTION ET TRAITEMENT DES OUTLIERS
# ============================================================================

print("="*60)
print("DÉTECTION DES OUTLIERS")
print("="*60)

# Visualisation de la distribution avec un histogramme
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(size, bins=20, edgecolor='black')
plt.xlabel('Taille du groupe')
plt.ylabel('Fréquence')
plt.title('Distribution de la taille des groupes')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# MÉTHODE DES 3 SIGMA (3 ÉCARTS-TYPES)
# ============================================================================

# Calcul des bornes selon la règle des 3 sigma
# Valeurs normales: moyenne ± 3 × écart-type
min_val = size.mean() - (3 * size.std())
max_val = size.mean() + (3 * size.std())

print(f"\nMéthode des 3 sigma:")
print(f"  Moyenne: {size.mean():.2f}")
print(f"  Écart-type: {size.std():.2f}")
print(f"  Borne inférieure (μ - 3σ): {min_val:.2f}")
print(f"  Borne supérieure (μ + 3σ): {max_val:.2f}")
print()

# Identification des outliers (valeurs au-delà des bornes)
outliers = size[size > max_val]
print(f"Nombre d'outliers détectés (> {max_val:.2f}): {outliers.count()}")
print()

# Affichage des outliers
if len(outliers) > 0:
    print("Valeurs des outliers:")
    print(outliers.values)
    print()

# SUPPRESSION des outliers
# On garde seulement les valeurs <= max_val
size_cleaned = size[size <= max_val]
print(f"Taille avant suppression: {size.shape[0]} échantillons")
print(f"Taille après suppression: {size_cleaned.shape[0]} échantillons")
print(f"Échantillons supprimés: {size.shape[0] - size_cleaned.shape[0]}")
print()

# Mise à jour de la variable
size = size_cleaned

# ============================================================================
# EXERCICE 1.03: ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================================================

print("="*60)
print("ENCODAGE DES VARIABLES CATÉGORIELLES")
print("="*60)

from sklearn.preprocessing import LabelEncoder

# Création d'un encodeur
enc = LabelEncoder()

# LABEL ENCODING: transformation texte → nombres
# Utile pour les variables ordinales ou binaires

# Encodage de 'sex' (Male/Female → 0/1)
# .astype('str') assure que toutes les valeurs sont des chaînes
tips["sex"] = enc.fit_transform(tips['sex'].astype('str'))
print("Encodage de 'sex': Male→0, Female→1")

# Encodage de 'smoker' (No/Yes → 0/1)
tips["smoker"] = enc.fit_transform(tips['smoker'].astype('str'))
print("Encodage de 'smoker': No→0, Yes→1")

# Encodage de 'day' (Thur/Fri/Sat/Sun → 0/1/2/3)
tips["day"] = enc.fit_transform(tips['day'].astype('str'))
print("Encodage de 'day': jours → nombres")

# Encodage de 'time' (Lunch/Dinner → 0/1)
tips["time"] = enc.fit_transform(tips['time'].astype('str'))
print("Encodage de 'time': Lunch→0, Dinner→1")
print()

# Affichage du dataset après encodage
print("Dataset après encodage:")
print(tips.head())
print()

# ============================================================================
# EXERCICE 1.04: NORMALISATION ET STANDARDISATION
# ============================================================================

print("="*60)
print("NORMALISATION ET STANDARDISATION")
print("="*60)

# ============================================================================
# NORMALISATION (Min-Max Scaling)
# ============================================================================
# Transforme les valeurs dans l'intervalle [0, 1]
# Formule: (x - min) / (max - min)

print("NORMALISATION (Min-Max Scaling):")
print("-" * 30)

# Application de la normalisation à tout le dataset
tips_normalized = (tips - tips.min()) / (tips.max() - tips.min())

print("Dataset normalisé (10 premières lignes):")
print(tips_normalized.head(10))
print()

# Vérification des plages
print("Vérification des plages après normalisation:")
print(f"  Minimum: {tips_normalized.min().min():.2f}")  # Devrait être 0
print(f"  Maximum: {tips_normalized.max().max():.2f}")  # Devrait être 1
print()

# ============================================================================
# STANDARDISATION (Z-Score Normalization)
# ============================================================================
# Centre les données (moyenne=0) et normalise l'écart-type (σ=1)
# Formule: (x - moyenne) / écart-type

print("STANDARDISATION (Z-Score):")
print("-" * 30)

# Application de la standardisation à tout le dataset
tips_standardized = (tips - tips.mean()) / tips.std()

print("Dataset standardisé (10 premières lignes):")
print(tips_standardized.head(10))
print()

# Vérification des statistiques
print("Vérification après standardisation:")
print(f"  Moyennes (devraient être ≈0):")
for col in tips_standardized.columns[:3]:  # Afficher 3 colonnes
    print(f"    {col}: {tips_standardized[col].mean():.6f}")
print(f"  Écarts-types (devraient être ≈1):")
for col in tips_standardized.columns[:3]:  # Afficher 3 colonnes
    print(f"    {col}: {tips_standardized[col].std():.6f}")
print()

# ============================================================================
# COMPARAISON NORMALISATION VS STANDARDISATION
# ============================================================================

print("="*60)
print("COMPARAISON DES MÉTHODES")
print("="*60)

print("""
NORMALISATION (Min-Max):
✓ Avantages:
  - Valeurs bornées dans [0, 1]
  - Préserve la distribution originale
  - Utile pour les algorithmes sensibles aux plages (réseaux de neurones)
✗ Inconvénients:
  - Sensible aux outliers
  - Peut comprimer les données si outliers présents

STANDARDISATION (Z-Score):
✓ Avantages:
  - Moins sensible aux outliers
  - Utile pour les algorithmes basés sur les distances (KNN, SVM)
  - Préserve la forme de la distribution
✗ Inconvénients:
  - Pas de bornes fixes
  - Peut avoir des valeurs négatives

CONSEIL:
- Utilisez la standardisation par défaut
- Utilisez la normalisation si vous avez besoin de valeurs dans [0, 1]
""")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("="*60)
print("RÉSUMÉ DE L'EXERCICE")
print("="*60)

print(f"""
Étapes de prétraitement réalisées:

1. VALEURS MANQUANTES:
   - {missing_count} valeurs manquantes créées
   - Imputées avec la moyenne ({mean})

2. OUTLIERS:
   - Détectés avec la méthode des 3 sigma
   - {outliers.count()} outliers trouvés et supprimés

3. ENCODAGE CATÉGORIEL:
   - 4 variables transformées (sex, smoker, day, time)
   - Label Encoding utilisé

4. NORMALISATION:
   - Min-Max: valeurs dans [0, 1]
   - Z-Score: moyenne=0, écart-type=1

Dataset prêt pour l'apprentissage automatique! 🎯
""")