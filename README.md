# 🏃‍♂️ Analyse et Prédiction - Marathon de Boston 2023

Ce projet analyse les données des coureurs du Marathon de Boston 2023 à l'aide de techniques de régression linéaire et d'exploration statistique, avec un focus sur la prédiction des temps finaux et l'étude du *positive split* (ralentissement en seconde partie de course).

## 🎯 Objectifs de l'étude
1. **Le temps au semi suffit-il pour prédire le temps final ?** (Modèle de régression simple)
2. **Ajouter l'âge et le sexe améliore-t-il la prédiction ?** (Modèle de régression multiple)
3. **Comment varie le *positive split* selon l'âge, le sexe et le niveau de course ?**

## 📊 Principaux Résultats
- **Prédiction précise :** Le temps au semi explique **95 % de la variance** du temps final ($R^2 = 0.95$), avec une erreur moyenne (MAE) d'environ 6,5 minutes.
- **Gain marginal :** Ajouter l'âge et le sexe n'apporte qu'une amélioration minime de la prédiction (gain d'environ 3 secondes sur l'erreur moyenne). Le temps intermédiaire contient déjà l'essentiel de l'information.
- **Gestion de l'effort :** **89 % des coureurs subissent un ralentissement** en seconde partie de course (médiane de 5,8 %). Ce ralentissement augmente avec l'âge, mais est surtout corrélé au niveau initial de performance.

## 🛠️ Stack Technique
- **Langage :** Python 3
- **Manipulation & Analyse :** Pandas, NumPy
- **Visualisation :** Matplotlib
- **Machine Learning :** Scikit-Learn (LinearRegression, K-Fold cross-validation)
- **Format :** MyST Markdown (Jupytext)

## 🚀 Installation et Utilisation
1. Cloner le dépôt :
   ```bash
   git clone [https://github.com/ton-pseudo/marathon-boston-2023.gits](https://github.com/ton-pseudo/marathon-boston-2023.gits)
