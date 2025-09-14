# ============================================================================
# EXERCICE 1.01 - SÉPARATION DES FEATURES ET DE LA CIBLE
# ============================================================================
# Cet exercice montre comment séparer un dataset en:
# - X: les features (variables indépendantes) pour prédire
# - y: la cible (variable dépendante) à prédire
# ============================================================================

# Importation des bibliothèques nécessaires
import seaborn as sns  # Pour charger des datasets d'exemple
import pandas as pd   # Pour manipuler les données sous forme de tableaux
import numpy as np   # Pour les calculs mathématiques

# ============================================================================
# CHARGEMENT DU DATASET
# ============================================================================

# Chargement du dataset 'tips' (pourboires dans un restaurant)
# Ce dataset contient: total_bill, tip, sex, smoker, day, time, size
tips = pd.read_csv("data/tips.csv")

# ============================================================================
# SÉPARATION FEATURES (X) ET CIBLE (y)
# ============================================================================

# X: Toutes les colonnes SAUF 'tip' (nos features/variables explicatives)
# .drop() supprime la colonne spécifiée
# axis=1 indique qu'on supprime une colonne (axis=0 pour les lignes)
X = tips.drop('tip', axis=1)

# Affichage des 10 premières lignes pour vérifier
print("Features (X) - Les variables utilisées pour la prédiction:")
print(X.head(10))
print()

# Affichage de la forme du tableau X
# .shape retourne (nombre_de_lignes, nombre_de_colonnes)
print(f"Dimensions de X: {X.shape}")
print(f"  → {X.shape[0]} échantillons (clients)")
print(f"  → {X.shape[1]} features (caractéristiques)")
print()

# ============================================================================
# EXTRACTION DE LA VARIABLE CIBLE
# ============================================================================

# Y: Uniquement la colonne 'tip' (notre cible/variable à prédire)
# On veut prédire le montant du pourboire
Y = tips['tip']

# Affichage des 10 premières valeurs cibles
print("Cible (Y) - La variable à prédire (montant du pourboire):")
print(Y.head(10))
print()

# Affichage de la forme du vecteur Y
print(f"Dimensions de Y: {Y.shape}")
print(f"  → {Y.shape[0]} valeurs cibles")
print()

# ============================================================================
# ANALYSE RAPIDE
# ============================================================================

print("="*60)
print("RÉSUMÉ DE LA SÉPARATION")
print("="*60)

print(f"""
Dataset: Tips (pourboires au restaurant)

FEATURES (X) - Variables explicatives:
- total_bill: Montant total de l'addition
- sex: Sexe du client
- smoker: Si le client fume ou non
- day: Jour de la semaine
- time: Moment (Lunch/Dinner)
- size: Nombre de personnes à table

CIBLE (Y) - Variable à prédire:
- tip: Montant du pourboire

Objectif: Prédire le pourboire en fonction des autres variables
Type de problème: Régression (prédire une valeur continue)

Prochaines étapes typiques:
1. Encoder les variables catégorielles (sex, smoker, day, time)
2. Normaliser les variables numériques
3. Diviser en train/test
4. Entraîner un modèle de régression
""")

# ============================================================================
# VÉRIFICATIONS IMPORTANTES
# ============================================================================

# Vérifier qu'on a le même nombre d'échantillons
assert X.shape[0] == Y.shape[0], "Erreur: X et Y doivent avoir le même nombre d'échantillons!"
print(f"✓ Vérification OK: X et Y ont bien {X.shape[0]} échantillons chacun")