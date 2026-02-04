# 📦 Retail Demand Forecasting — LSTM vs LightGBM (End-to-End)



Prévision de la demande (ventes quotidiennes) sur un jeu de données retail multi-séries (\*\*store × item\*\*).  

Le dépôt contient un pipeline complet : préparation des données, création de fenêtres temporelles, entraînement d’un modèle séquentiel (LSTM), baselines tabulaires, évaluation et génération de figures.



---



## 🗂️ Dataset

Kaggle — \*Store Item Demand Forecasting Challenge\*  

Colonnes : `date`, `store`, `item`, `sales`



Place `train.csv` ici :



```text

data/train.csv
````


## 🧱 Project Structure



```text

lstm-demand-forecasting/

├── configs/

│   └── default.yaml

├── src/

│   ├── data/

│   │   ├── make\_dataset.py

│   │   └── windowing.py

│   ├── models/

│   │   ├── baselines.py

│   │   ├── lgbm\_baseline.py

│   │   └── lstm\_model.py

│   ├── train.py

│   ├── evaluate.py

│   ├── predict.py

│   └── utils.py

├── tests/

│   └── test\_windowing.py

├── artifacts/               # generated after training

├── reports/

│   └── figures/             # generated plots

├── requirements.txt

└── README.md
````


---



## ⚙️ Installation



```bash

python -m venv .venv

# Windows

.venv\\Scripts\\activate

# Mac/Linux

source .venv/bin/activate



pip install -r requirements.txt

````

---



## 🚀 Utilisation



### 1) Entraîner le modèle LSTM

```bash

python -m src.train --config configs/default.yaml

````



Fichiers générés :



\* `artifacts/model.keras`

\* `artifacts/meta.json` (métadonnées + scaler)



### 2) Évaluer et comparer les modèles



```bash

python -m src.evaluate --config configs/default.yaml

```



Fichier généré :



* `reports/figures/forecast\_comparison.png`



### 3) Produire une prédiction (ex: J+90)



```bash

python -m src.predict --config configs/default.yaml --store 1 --item 1 --start-date 2017-10-01 --horizon-days 90

```



Fichier généré :



\* `artifacts/forecast.csv`



---



## 🧠 Approche



### Fenêtrage (supervisé)



Pour chaque série `(store, item)` :



\* \*\*Entrée\*\* : 28 jours d’historique (`lookback=28`)

\* \*\*Sortie\*\* : 7 jours à prédire (`horizon=7`)



Features calendaires ajoutées : `day-of-week`, `month`, `is\_weekend`, `day`.



---



## 🧩 Modèles



### Baselines



\* \*\*Naive(last)\*\* : répète la dernière valeur observée

\* \*\*MA(7)\*\* : moyenne mobile sur 7 jours

\* \*\*LightGBM (lags)\*\* : features de l’historique (`last`, `mean7`, `mean14`, `std7`, `trend`) + apprentissage direct multi-horizon



### LSTM (TensorFlow/Keras)



Modèle séquentiel global (un seul modèle pour toutes les séries) :



\* `sales\_seq` (standardisée)

\* `cal\_seq` (features calendaires)

\* embeddings `store\_id` / `item\_id` pour capturer les effets spécifiques à chaque série



Sortie : vecteur de taille 7 (forecast multi-step).



---



## 📊 Résultats (test split)



| Modèle                     |       MAE |      RMSE |     sMAPE |

| -------------------------- | --------: | --------: | --------: |

| Naive(last)                |    54.677 |    61.302 |     1.975 |

| MA(7)                      |    54.661 |    61.288 |     1.976 |

| LightGBM (lags)            |     8.415 |    11.221 |     0.169 |

| LSTM (global + embeddings) | \*\*6.214\*\* | \*\*8.185\*\* | \*\*0.129\*\* |



---



## 🖼️ Visualisations



Les figures sont générées automatiquement lors de l’évaluation et sauvegardées dans `reports/figures/`.



### 1) Comparaison des modèles (horizon 7 jours)

\*\*Figure :\*\* `reports/figures/forecast\_comparison.png`  

Comparaison sur un exemple aléatoire :

\- \*\*True\*\* : ventes réelles sur les 7 jours à prédire  

\- \*\*LSTM\*\* : prévision multi-horizon (7 jours)  

\- \*\*Baselines\*\* : Naive(last), MA(7), LightGBM (lags)



> Objectif : visualiser rapidement l’écart entre les modèles, et repérer les cas où le LSTM capte (ou non) la dynamique.



!\[Forecast comparison](reports/figures/forecast\_comparison.png)



### 2) Métriques d’évaluation (console)

Lors de `python -m src.evaluate ...`, le script affiche :

\- \*\*MAE\*\* (Mean Absolute Error)

\- \*\*RMSE\*\* (Root Mean Squared Error)

\- \*\*sMAPE\*\* (Symmetric Mean Absolute Percentage Error)



Ces métriques permettent de comparer les approches sur le split test, avec une lecture “business” (erreur absolue moyenne) et une lecture “stabilité” (RMSE).



---



## 🧪 Tests



```bash

pytest -q

```



---



## 🔧 Configuration



Les principaux paramètres sont dans `configs/default.yaml` :



\* `lookback`, `horizon`

\* split temporel (`train\_end`, `val\_end`, `test\_end`)

\* hyperparamètres (`epochs`, `batch\_size`, `lstm\_units`, `dropout`...)



---



## 🔭 Next steps (roadmap)



\- Ajouter des modèles SOTA de forecasting (ex: TFT / PatchTST / N-BEATS) et comparer aux baselines.

\- Prédiction probabiliste (intervalles P50/P90) pour quantifier l’incertitude.

\- Monitoring : drift sur la distribution des ventes et recalibrage périodique.

\- Feature store léger : lags + événements (promotions, jours fériés) quand disponibles.





