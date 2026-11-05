---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

<div style="background-color:#d1ecf1; border-left:6px solid #17a2b8; padding:8px;">
<b> Marathon de Boston 2023
</div>

+++

Questions posées :
1. Le temps au semi suffit-il pour prédire le temps final ?
2. Ajouter l'âge et le sexe améliore-t-il la prédiction ?
3. Comment varie le "positive split" (ralentissement) selon l'âge et le sexe ?

+++

<div style="background-color: #F08975 ; border-left:6px solid #F54927; padding:8px;">
<b> 1. Initialisation et chargement des données
</div>

```{code-cell}
# Librairies
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RANDOM_STATE = 42
```

```{code-cell}
# Chargement des données
FILE_PATH = "boston_marathon_2023.csv"
df = pd.read_csv(FILE_PATH)
print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
df.head()
```

```{code-cell}
df.info()
```

```{code-cell}
df.describe()
```

<div style="background-color: #F08975 ; border-left:6px solid #F54927; padding:8px;">
<b> 2. Exploration des données et nettoyage
</div>

```{code-cell}
# Détection des valeurs vides et des doublons
print("Valeurs manquantes par colonne :")
print(df.isna().sum()[df.isna().sum() > 0])
print(f"\nDoublons : {df.duplicated().sum()}")
```

```{code-cell}
# Nettoyage : on garde les coureurs avec temps au semi et temps final valides car on veut prédire le temps final à partir du temps au semi
df_clean = df.dropna(subset=["half_time_sec", "finish_net_sec"])
print(f"Lignes conservées : {len(df_clean)} / {len(df)}")
```

<div style="background-color: #F08975 ; border-left:6px solid #F54927; padding:8px;">
<b> 3. Exploration graphique
</div>

```{code-cell}
# Boîtes à moustaches par groupe d'âge
# Tri "naturel" des groupes d'âge si format '18-39', '40-44', etc.
def sort_key_age(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 9999

age_groups = sorted(df_clean["age_group"].unique(), key=sort_key_age)
box_data_age = [df_clean.loc[df_clean["age_group"] == ag, "finish_net_minutes"].values for ag in age_groups]

plt.figure(figsize=(10, 5))
plt.boxplot(box_data_age, tick_labels=age_groups, showfliers=True)
plt.title("Temps final net (min) – par groupe d’âge")
plt.ylabel("Minutes")
plt.xlabel("Groupe d’âge")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
```

Une tendance se dessine : lorsque l'âge augmente, le temps au marathon semble augmenter aussi.

```{code-cell}
plt.figure(figsize=(8, 6))
plt.scatter(df_clean["half_time_sec"], df_clean["finish_net_sec"], s=6, alpha=0.4)
plt.plot(df_clean["half_time_sec"], 2*df_clean["half_time_sec"], linewidth=2, color="red")
plt.xlabel("Semi-marathon (s)")
plt.ylabel("Temps final net (s)")
plt.title("Marathon de Boston 2023")
plt.grid(True)
plt.tight_layout()
plt.show()
```

La droite rouge représente un temps identique sur la première et la seconde moitié. Les coureurs les plus proches de cette ligne sont les plus réguliers. Au-dessus de la ligne, les coureurs ont mis plus de temps sur la seconde partie que la première. En dessous de la ligne, ils sont allés plus vite sur la seconde partie que la première. Les meilleurs coureurs sont en bas à gauche du graphique.

+++

<div style="background-color: #F08975 ; border-left:6px solid #F54927; padding:8px;">
<b> 4. Modélisation : prédiction du temps final à partir du temps au semi
</div>

+++

Le temps au semi suffit-il pour prédire le temps final ?

```{code-cell}
# Sélection caractéristique et variable cible
features = ["half_time_sec"]
target = "finish_net_sec"
```

```{code-cell}
# Séparation des données (caractéristiques X et valeurs cibles y)
X = df_clean[features]
y = df_clean[target]
```

```{code-cell}
# Séparation des données en 80% de données d'entraînement et 20% de données de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
```

```{code-cell}
# Pas de standardisation : inutile ici, et le coefficient reste directement interprétable
# Choix du modèle
model = LinearRegression()
```

```{code-cell}
# Entrainement du modèle
model.fit(X_train, y_train)
```

```{code-cell}
# Evaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print(f"R² : {r_squared:.3f} | MAE : {mae/60:.1f} min | RMSE : {rmse/60:.1f} min")
print(f"Coefficient : {model.coef_[0]:.3f}")
print(f"Ordonnée : {model.intercept_:.3f}")
```

```{code-cell}
# Validation croisée à 5 plis
scoring = {
    'R2': 'r2',
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error'
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = cross_validate(model, X, y, cv=cv, scoring=scoring)

df_scores = pd.DataFrame({
    'R2': results['test_R2'],
    'MAE': -results['test_MAE'],
    'RMSE': -results['test_RMSE']
})

df_scores.index = [f'Pli {i+1}' for i in range(len(df_scores))]
df_scores.loc['Moyenne'] = df_scores.mean()

print(df_scores.round(4))
```

```{code-cell}
print(f"R² moyen : {results['test_R2'].mean():.4f}")
print(f"MAE moyen : {-results['test_MAE'].mean() / 60:.2f} min")
```

```{code-cell}
valeur_a_predire = pd.DataFrame({'half_time_sec': [6000]})
temps_final = model.predict(valeur_a_predire) [0]
print(f"Temps final prédit : {temps_final:.0f} s ({temps_final/ 3600:.2f} h)")
```

Le temps au semi explique 95 % de la variance du temps final (R² = 0,95), avec une erreur moyenne de 6,6 minutes (RMSE : 9,8 minutes, ce qui indique quelques coureurs nettement moins bien prédits). Le coefficient de 2,22 signifie qu’un coureur plus lent de 1 minute au semi finit en moyenne 2,2 minutes plus lent : l’écart se creuse en seconde moitié, surtout chez les coureurs les plus lents. La validation croisée donne des résultats stables (R² entre 0,951 et 0,957), donc le modèle généralise bien. Il reste toutefois une part d’erreur que le seul temps au semi ne peut pas expliquer.

+++

<div style="background-color: #F08975 ; border-left:6px solid #F54927; padding:8px;">
<b> 5. Modélisation : prédiction du temps final à partir d'autres variables
</div>

+++

Ajouter l'âge et le sexe améliore-t-il la prédiction ?

```{code-cell}
# Sélection caractéristiques, la variable cible n'a pas changée
features_multi = ["half_time_sec", "age_group", "gender"]

X_multi = pd.get_dummies(
    df_clean[features_multi],
    columns=["age_group", "gender"],
    drop_first=True,
    dtype=int,
)

X_multi.head()
```

```{code-cell}
# Séparation des données en 80% de données d'entraînement et 20% de données de test
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=RANDOM_STATE)
```

```{code-cell}
# Choix du modèle
model_multi = LinearRegression()
```

```{code-cell}
# Entraînement du modèle
model_multi.fit(X_train_m, y_train_m)
```

```{code-cell}
# Evaluation du modèle
y_pred_m = model_multi.predict(X_test_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
rmse_m = np.sqrt(mse_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)
print(f"R² : {r2_m:.3f} | MAE : {mae_m/60:.1f} min | RMSE : {rmse_m/60:.1f} min")
print(f"Coefficient : {model_multi.coef_[0]:.3f}")
print(f"Coefficient : {model_multi.intercept_:.3f}")
```

```{code-cell}
results_m = cross_validate(LinearRegression(), X_multi, y, cv=cv, scoring=scoring)

df_scores_m = pd.DataFrame({
    'R2': results_m['test_R2'],
    'MAE': -results_m['test_MAE'],
    'RMSE': -results_m['test_RMSE']
})

df_scores_m.index = [f'Pli {i+1}' for i in range(len(df_scores_m))]
df_scores_m.loc['Moyenne'] = df_scores_m.mean()

print(df_scores_m.round(4))
```

```{code-cell}
print(f"R² moyen : {results_m['test_R2'].mean():.4f}")
print(f"MAE moyen : {-results_m['test_MAE'].mean() / 60:.2f} min")
```

```{code-cell}
comparaison = pd.DataFrame({
    "Simple (temps au semi)": [r_squared, mae / 60, rmse / 60],
    "Multiple (temps au semi + âge + sexe)": [r2_m, mae_m / 60, rmse_m / 60],
}, index=["R²", "MAE (min)", "RMSE (min)"]).round(3)
print(comparaison)
```

Ajouter l’âge et le sexe améliore-t-il la prédiction ? Très peu ! Le modèle multiple obtient un R² de 0,952 contre 0,951 pour le modèle simple, et une erreur moyenne de 6,5 minutes contre 6,6 minutes, soit un gain d’environ 3 secondes. Le temps au semi contient donc déjà presque toute l’information nécessaire pour prédire le temps final. L’âge et le sexe n’apportent qu’un gain marginal, qui ne justifie pas à lui seul d’ajouter ces variables pour la prédiction.

```{code-cell}
coefs = pd.Series(model_multi.coef_, index=X_multi.columns)
print(f"half_time_sec : {coefs['half_time_sec']:.3f}  (sans unité : secondes finales par seconde au semi)")
print("\nEffets de l'âge et du sexe, en minutes :")
print((coefs.drop("half_time_sec") / 60).round(1))
```

À temps au semi égal, les femmes finissent en moyenne 3,1 minutes plus vite que les hommes, et les écarts entre groupes d’âge restent inférieurs à 2 minutes (hors 80 ans et plus, groupe de 5 coureurs).

+++

<div style="background-color: #F08975 ; border-left:6px solid #F54927; padding:8px;">
<b> 6. Influence de l'âge et du sexe sur le second semi
</div>

+++

Comment varie le "positive split" (ralentissement) selon l'âge et le sexe ?

```{code-cell}
# Variations par classe d'âge
n_groups = len(age_groups)

# Taille de la figure : ajuste dynamiquement selon le nombre de catégories
fig, axes = plt.subplots(
    nrows=(n_groups + 2) // 3,
    ncols=min(3, n_groups),
    figsize=(15, 5 * ((n_groups + 2) // 3))
)

axes = axes.flatten()

for i, age in enumerate(age_groups):
    ax = axes[i]
    subset = df_clean[df_clean["age_group"] == age]

    ax.scatter(subset["half_time_sec"], subset["finish_net_sec"], s=6, alpha=0.4)
    ax.plot(subset["half_time_sec"], 2 * subset["half_time_sec"], linewidth=2, color="red")

    ax.set_xlabel("Semi-marathon (s)")
    ax.set_ylabel("Temps final net (s)")
    ax.set_title(f"Groupe d'âge : {age}")
    ax.grid(True)

# Supprimer les sous-graphiques vides s'il y en a
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Marathon de Boston 2023 — par groupe d'âge", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
```

```{code-cell}
df_clean = df_clean.copy()

# Temps de la seconde moitié
df_clean["second_half_sec"] = df_clean["finish_net_sec"] - df_clean["half_time_sec"]

# Ralentissement en % : positif = plus lent en seconde moitié, négatif = negative split
df_clean["slowdown_pct"] = (df_clean["second_half_sec"] / df_clean["half_time_sec"] - 1) * 100

df_clean["slowdown_pct"].describe()
```

```{code-cell}
part_positive = (df_clean["slowdown_pct"] > 0).mean() * 100
print(f"{part_positive:.1f} % des coureurs ralentissent en seconde moitié")
print(f"Ralentissement médian : {df_clean['slowdown_pct'].median():.1f} %")
```

```{code-cell}
par_sexe = df_clean.groupby("gender")["slowdown_pct"].agg(["count", "median", "mean"]).round(2)
print(par_sexe)
```

```{code-cell}
par_age = (
    df_clean.groupby("age_group")["slowdown_pct"]
    .agg(["count", "median", "mean"])
    .loc[age_groups]
    .round(2)
)
print(par_age)
```

```{code-cell}
croise = df_clean.groupby(["age_group", "gender"])["slowdown_pct"].median().unstack("gender").loc[age_groups]

ax = croise.plot(kind="bar", figsize=(10, 5))
ax.set_title("Ralentissement médian en seconde moitié, par âge et par sexe")
ax.set_xlabel("Groupe d'âge")
ax.set_ylabel("Ralentissement médian (%)")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(True, axis="y", alpha=0.3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
```

```{code-cell}
# Ralentissement médian par âge, à niveau comparable (quartiles du temps au semi)
df_clean["niveau_semi"] = pd.qcut(
    df_clean["half_time_sec"], 4, labels=["Q1 (rapides)", "Q2", "Q3", "Q4 (lents)"]
)
print(
    df_clean.groupby(["age_group", "niveau_semi"], observed=True)["slowdown_pct"]
    .median().unstack("niveau_semi").loc[age_groups].round(2)
)

# les cases vides signifient qu’aucun coureur du groupe n’appartient à ce quartile
```

```{code-cell}
# Pourquoi la moyenne diffère-t-elle entre hommes et femmes ?
print(df_clean.groupby("gender")["slowdown_pct"]
      .describe(percentiles=[.25, .5, .75, .9]).round(2))
```

89 % des coureurs ralentissent en seconde moitié, avec un ralentissement médian de 5,8 %. Ce ralentissement augmente avec l’âge (de 4,8 % chez les 18-39 ans à 8,5 % chez les 65-69 ans, en excluant les groupes avec peu de coureurs), mais cette hausse s’explique en grande partie par le niveau de course : à niveau comparable, les médianes se rapprochent, et elles sont quasiment identiques entre groupes d’âge chez les coureurs les plus lents (environ 10 %). Chez les coureurs rapides, un écart modéré persiste (de 3,7 % chez les 18-39 ans à 6,2 % chez les 60-64 ans). Le ralentissement médian est identique chez les hommes et les femmes (5,8 %), mais les hommes sont plus dispersés : 10 % d’entre eux ralentissent de plus de 19,7 %, contre 16,8 % chez les femmes. Les groupes de 75 ans et plus comptent peu de coureurs et les résultats les concernant sont à interpréter avec prudence. Ces résultats décrivent des tendances sur une seule course et n’ont pas fait l’objet de tests statistiques.
