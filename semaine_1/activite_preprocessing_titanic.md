# Activité : Préparation des Données du Titanic

## Contexte
Vous travaillez pour le département de sécurité d'une compagnie de croisière. Suite à une analyse historique de l'accident du Titanic, votre département souhaite développer un modèle prédictif pour améliorer les protocoles de sécurité. Votre mission est de préparer le dataset du Titanic pour l'entraînement d'un modèle de machine learning.

## Objectif
Préparer complètement le dataset du Titanic en utilisant toutes les techniques de prétraitement que vous avez apprises. À la fin de cette activité, vous devez avoir un dataset prêt pour l'entraînement d'un modèle.

## Dataset
Le dataset du Titanic contient les informations suivantes sur les passagers :
- **survived** : Survie (0 = Non, 1 = Oui) - *Variable cible*
- **pclass** : Classe du billet (1 = 1ère, 2 = 2ème, 3 = 3ème)
- **sex** : Sexe
- **age** : Âge
- **sibsp** : Nombre de frères/sœurs/époux à bord
- **parch** : Nombre de parents/enfants à bord
- **fare** : Tarif du billet
- **embarked** : Port d'embarquement (C = Cherbourg, Q = Queenstown, S = Southampton)
- **class** : Classe du billet (catégorielle)
- **who** : Homme, Femme ou Enfant
- **adult_male** : Indicateur booléen
- **deck** : Pont du bateau
- **embark_town** : Nom de la ville d'embarquement
- **alive** : Si le passager a survécu
- **alone** : Si le passager voyageait seul

## Instructions

### Étape 1 : Chargement des données
1. Importez les bibliothèques nécessaires pour l'analyse de données et le prétraitement
2. Chargez le dataset du Titanic (disponible via `seaborn.load_dataset('titanic')` ou via un fichier CSV avec Pandas)
3. Explorez rapidement le dataset pour comprendre sa structure

### Étape 2 : Sélection des features
1. Analysez toutes les colonnes disponibles
2. Identifiez les features redondantes ou qui apportent la même information
3. **Décision métier importante** : Sélectionnez les features les plus pertinentes pour prédire la survie
   - Justifiez vos choix dans un commentaire
   - Pensez à la redondance entre certaines variables

### Étape 3 : Analyse des données manquantes
1. Identifiez toutes les valeurs manquantes dans votre dataset
2. Pour chaque feature avec des valeurs manquantes :
   - Analysez le pourcentage de données manquantes
   - Choisissez une stratégie appropriée :
     - Suppression (si peu de lignes concernées)
     - Imputation par la moyenne/médiane (variables numériques)
     - Imputation par le mode (variables catégorielles)
     - Autres méthodes plus sophistiquées
3. Documentez et justifiez vos choix

### Étape 4 : Détection et traitement des outliers
1. Pour chaque variable numérique :
   - Visualisez la distribution (histogramme, boxplot)
   - Identifiez les outliers potentiels
   - Décidez s'il faut les traiter et comment :
     - Suppression
     - Capping (limitation aux valeurs extrêmes)
     - Transformation (log, racine carrée)
     - Conservation (si les outliers sont légitimes)

### Étape 5 : Encodage des variables catégorielles
1. Identifiez toutes les variables catégorielles (texte)
2. Pour chaque variable catégorielle, choisissez la méthode d'encodage appropriée :
   - Label Encoding (pour les variables ordinales)
   - One-Hot Encoding (pour les variables nominales)
   - Autre méthode si justifiée
3. Appliquez l'encodage choisi

### Étape 6 : Mise à l'échelle des données
1. Analysez la distribution et l'échelle de vos variables numériques
2. Choisissez une méthode de mise à l'échelle :
   - Normalisation (Min-Max Scaling)
   - Standardisation (Z-score)
   - Autre méthode
3. Appliquez la transformation choisie
4. **Important** : Réfléchissez à quel moment appliquer cette transformation (avant ou après le split train/test)

### Étape 7 : Vérification finale
1. Vérifiez que votre dataset final :
   - N'a plus de valeurs manquantes
   - N'a plus de variables textuelles
   - A des features à échelle comparable
   - Est prêt pour l'entraînement d'un modèle

## Livrables attendus

1. **Code Python** : Un notebook Jupyter ou script Python commenté avec toutes les étapes
2. **Rapport d'analyse** : Un document expliquant :
   - Vos choix de features et leur justification
   - Les méthodes utilisées pour traiter les valeurs manquantes
   - La stratégie de gestion des outliers
   - Les techniques d'encodage choisies
   - La méthode de mise à l'échelle sélectionnée
3. **Dataset final** : Le dataset prétraité sauvegardé en CSV


## Conseils

- Prenez le temps d'explorer les données avant de prendre des décisions
- Documentez chaque étape et chaque choix
- Pensez à l'impact de vos décisions sur le modèle final
- N'hésitez pas à tester plusieurs approches et comparer les résultats
- Gardez une trace de la forme originale des données pour pouvoir revenir en arrière si nécessaire

## Questions de réflexion

1. Pourquoi certaines features sont-elles redondantes ?
2. Quel impact le traitement des valeurs manquantes peut-il avoir sur le modèle ?
3. Les outliers sont-ils toujours mauvais ? Dans quel contexte pourrait-on vouloir les conserver ?
4. Quelle est la différence entre normalisation et standardisation ? Quand utiliser l'une ou l'autre ?
5. Pourquoi est-il important de faire la mise à l'échelle après le split train/test ?



**Prérequis** : 
- Connaissance de base de Python
- Compréhension des concepts de preprocessing vus en cours
- Familiarité avec pandas et numpy