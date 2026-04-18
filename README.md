# 🏠 House Price Predictor — ML Project

> Prédiction du prix des maisons en Californie à l'aide du Machine Learning, avec une interface interactive Gradio.

---

## 📌 Description

Ce projet utilise des algorithmes de **Machine Learning** pour prédire le prix moyen d'une maison en Californie, en se basant sur des caractéristiques physiques du logement (superficie, âge, occupation) et des données géographiques obtenues via des APIs externes (Geocod.io).

L'interface utilisateur est construite avec **Gradio**, permettant une interaction simple et rapide sans écrire de code.

---

## ⚠️ Note importante — Modèles non inclus dans ce dépôt

> **Les fichiers du dossier `model/` (`RandomForest.joblib`, `GeoAxisModel.joblib`, `Scaler.joblib`) ne sont pas inclus dans ce dépôt GitHub.**

**Raison :** Ces fichiers binaires sérialisés dépassent la limite de taille imposée par GitHub (**100 MB**), ce qui les rend incompatibles avec un dépôt Git standard. Leur inclusion aurait alourdi inutilement le dépôt et dégradé les performances de versionnement.

**Solution :** Pour utiliser l'application, vous devez **régénérer les modèles localement** en exécutant le notebook d'entraînement fourni :

```bash
# Ouvrir et exécuter dans l'ordre toutes les cellules du notebook
notebook/Projet_Ml.ipynb
```

Les modèles seront alors sauvegardés automatiquement dans le dossier `model/`.

> 💡 Pour les projets futurs nécessitant le versionnement de grands fichiers binaires, il est recommandé d'utiliser [Git LFS](https://git-lfs.github.com/) ou un stockage externe (Google Drive, HuggingFace Hub, AWS S3, etc.).

---

## 🗂️ Structure du Projet

```
PROJET ML_house pricing/
│
├── interface/
│   └── app.py                  # Interface Gradio + logique de prédiction
│
├── model/                      # ⚠️ Non versionné (taille > 100 MB)
│   ├── RandomForest.joblib     # Modèle Random Forest entraîné
│   ├── Scaler.joblib           # Normalisateur StandardScaler
│   └── GeoAxisModel.joblib     # Modèle de clustering géographique (KMeans)
│
├── notebook/
│   └── Projet_Ml.ipynb         # Notebook d'exploration, entraînement et évaluation
│
├── .gitignore                  # Exclusion du dossier model/
└── README.md
```

---

## ⚙️ Fonctionnement

1. L'utilisateur saisit les caractéristiques de la maison dans l'interface Gradio.
2. L'adresse (ville + quartier) est convertie en coordonnées GPS via l'**API Geocod.io**.
3. Des features dérivées sont calculées (logs, ratios, interactions).
4. Un modèle **GeoAxis** (KMeans) encode la localisation géographique.
5. Le modèle **Random Forest** prédit le prix (en log), puis le résultat est retransformé en valeur réelle.

---

## 🧠 Modèles & Features

### Features utilisées

| Feature | Description |
|---|---|
| `MedInc` | Revenu médian du quartier |
| `HouseAge` | Âge de la maison (en années) |
| `AveRooms` | Nombre moyen de pièces |
| `AveBedrms` | Nombre moyen de chambres |
| `Population` | Population du bloc |
| `AveOccup` | Occupation moyenne |
| `Latitude` / `Longitude` | Coordonnées GPS |
| `GeoAxis` | Cluster géographique (calculé par KMeans) |
| `MedInc_log1p` | Log du revenu médian |
| `Rooms_per_occupant` | Ratio pièces/occupant |
| `Bedrooms_per_room` | Ratio chambres/pièces |
| `MedInc_times_GeoAxis` | Interaction revenu × cluster géo |

### Pipeline de modélisation

```
Input → Feature Engineering → StandardScaler → Random Forest → log-inverse → Prix ($)
```

---

## 🖼️ Interface Gradio

L'application expose une interface web locale avec les champs suivants :

- **Nombre de pièces** (Rooms)
- **Nombre de chambres à coucher** (Bedrooms)
- **Âge de la maison** (House Age)
- **Occupation** (AveOccup)
- **Ville** (ex: `Los Angeles, CA`)
- **Quartier** (ex: `Hollywood`)

Après avoir cliqué sur **"Enregistrer et prédire"**, l'application retourne le prix estimé en dollars.

---

## 🚀 Installation & Lancement

### Prérequis

- Python 3.8+
- pip

### 1. Cloner le dépôt

```bash
git clone https://github.com/aichasaaydi-ILIA/Prediction_house-pricing.git
cd Prediction_house-pricing
```

### 2. Installer les dépendances

```bash
pip install gradio joblib scikit-learn pandas numpy requests
```

### 3. Générer les modèles

Ouvrir et exécuter le notebook `notebook/Projet_Ml.ipynb` dans son intégralité. Cela crée les fichiers dans `model/`.

### 4. Lancer l'application

```bash
cd interface
python app.py
```

L'interface sera accessible sur : **http://127.0.0.1:7860**

---

## 📡 APIs utilisées

| API | Usage |
|---|---|
| [Geocod.io](https://www.geocod.io/) | Conversion adresse → coordonnées GPS (lat/lng) |
| US Census Bureau | Données démographiques (population, revenu médian) |

> ⚠️ Les clés API sont actuellement définies dans `app.py`. Pour un usage en production, il est recommandé de les stocker dans un fichier `.env` et de ne pas les versionner.

---

## 📊 Notebook d'entraînement

Le notebook `notebook/Projet_Ml.ipynb` contient :

- Exploration et visualisation des données (dataset California Housing de scikit-learn)
- Preprocessing et feature engineering
- Entraînement et comparaison de modèles
- Évaluation des performances (RMSE, R²)
- Sauvegarde des modèles avec `joblib`

---

## 🛠️ Technologies utilisées

| Technologie | Rôle |
|---|---|
| `scikit-learn` | Modèles ML, preprocessing |
| `Random Forest` | Modèle de prédiction principal |
| `joblib` | Sérialisation des modèles |
| `Gradio` | Interface utilisateur web |
| `pandas` / `numpy` | Manipulation des données |
| `requests` | Appels API externes |

---

## 👤 Auteur

Projet réalisé dans le cadre d'un cours de Machine Learning.

---

## 📄 Licence

Ce projet est à usage éducatif.
