# Cours: Introduction � l'Intelligence Artificielle et Machine Learning

## Description
Mat�riel p�dagogique complet pour un cours d'Intelligence Artificielle et Machine Learning destin� aux �tudiants de troisi�me ann�e, comprenant des pr�sentations interactives et des travaux pratiques.

## Auteur
**Joseph Azar**  
Ma�tre de conf�rences  
Universit� Marie Louis Pasteur  
Chercheur � Femto-ST  
Email: joseph.azar@univ-fcomte.fr

## Contenu du Cours

### Pr�sentations RevealJS

#### 1. Introduction � l'Intelligence Artificielle (`presentation.html`)
- Histoire de l'IA (1950-pr�sent)
- Les pionniers et leurs contributions
- Taxonomie de l'IA moderne
- L'apprentissage automatique et ses paradigmes
- La r�volution du Deep Learning
- Les Transformers et Large Language Models
- L'IA g�n�rative
- Applications et perspectives futures

#### 2. Machine Learning et Pr�paration des Donn�es (`ml-preprocessing-presentation.html`)
- Fondamentaux du Machine Learning
- Types de variables et donn�es
- Feature Engineering
- Techniques d'encodage (One-Hot, Label Encoding)
- Normalisation et Standardisation
- Gestion des valeurs manquantes et outliers
- M�triques d'�valuation
- Validation crois�e

### Travaux Pratiques (`semaine_1/`)

#### Exercices de d�monstration
- **exercise1_01.py** : Introduction au preprocessing
- **exercise1_02.py** : Techniques avanc�es de pr�paration
- **exercise1_03.py** : Pipeline complet de preprocessing

#### Activit� pratique
- **activite_preprocessing_titanic.md** : Activit� compl�te sur le dataset Titanic
  - S�lection de features
  - Traitement des valeurs manquantes
  - D�tection et gestion des outliers
  - Encodage des variables cat�gorielles
  - Mise � l'�chelle des donn�es

## Installation

### Pr�requis
```bash
# Cr�er un environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows

# Installer les d�pendances
pip install -r requirements.txt
```

### Packages requis
- pandas (2.2.3)
- numpy (1.26.4)
- matplotlib (3.9.2)
- seaborn (0.13.2)
- scikit-learn (1.5.2)
- jupyter (1.0.0)
- scipy (1.14.1)
- statsmodels (0.14.4)

## Utilisation des Pr�sentations

### Visualisation
Ouvrez les fichiers HTML dans un navigateur web moderne:
- `presentation.html` pour l'introduction � l'IA
- `ml-preprocessing-presentation.html` pour le Machine Learning

### Navigation
- **Fl�ches** : Naviguer entre les slides
- **ESC/O** : Vue d'ensemble
- **F** : Plein �cran
- **S** : Notes du pr�sentateur

### Export PDF
Pour exporter en PDF, ajoutez `?print-pdf` � l'URL:
```
file:///chemin/vers/presentation.html?print-pdf
```
Puis utilisez la fonction d'impression du navigateur (Ctrl+P ou Cmd+P) et sauvegardez en PDF.

## Structure du Projet
```
ELC2-Intro-AI/
├── presentation.html              # Pr�sentation IA
├── ml-preprocessing-presentation.html  # Pr�sentation ML & Preprocessing
├── logo-umlp.png                 # Logo universit�
├── requirements.txt              # D�pendances Python
├── venv/                        # Environnement virtuel
├── semaine_1/                   # Travaux pratiques semaine 1
│   ├── exercise1_01.py
│   ├── exercise1_02.py
│   ├── exercise1_03.py
│   └── activite_preprocessing_titanic.md
└── README.md                    # Ce fichier

```

## Technologies utilis�es
- RevealJS 5.0.1 pour les pr�sentations
- Python 3.10+ pour les travaux pratiques
- HTML5/CSS3/JavaScript
- Biblioth�ques de Data Science (pandas, numpy, scikit-learn)

## Notes P�dagogiques
- Les pr�sentations sont con�ues pour �tre techniques et acad�miques tout en restant accessibles
- Les concepts math�matiques sont expliqu�s avec des exemples concrets
- Les travaux pratiques permettent une application imm�diate des concepts
- L'activit� Titanic encourage la r�flexion et les d�cisions m�tier

## Licence
Mat�riel p�dagogique � usage acad�mique - Universit� Marie Louis Pasteur