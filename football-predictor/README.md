# Pronostic IA — Prédiction de matchs de football

Application web de prédiction de résultats de matchs de football basée sur le Machine Learning. L'utilisateur sélectionne un match et obtient, en un clic, une analyse complète : probabilités 1N2, score probable, BTTS, Over/Under 2.5 buts et facteurs déterminants.

## Architecture

```
football-predictor/
├── backend/     FastAPI + scikit-learn/XGBoost/LightGBM (API de prédiction)
└── frontend/    Next.js 16 + Tailwind CSS (interface mobile-first)
```

Le frontend et le backend sont déployés **séparément** :

| Composant | Technologie | Hébergement recommandé |
|---|---|---|
| Frontend | Next.js (App Router) | **Vercel** |
| Backend  | FastAPI + modèles ML | **Render** ou **Railway** (gratuit) |

> **Pourquoi pas tout sur Vercel ?** Les fonctions serverless de Vercel ne sont pas adaptées à un backend Python chargeant des modèles scikit-learn/XGBoost/LightGBM en mémoire (taille de déploiement et durée d'exécution limitées). Render et Railway exécutent FastAPI comme un vrai serveur persistant, ce qui est nécessaire ici. Le frontend, lui, est fait pour Vercel.

## Fonctionnement

1. L'utilisateur choisit une compétition (Premier League, Liga, Bundesliga, Serie A, Ligue 1, Ligue des Champions, etc.)
2. Il sélectionne un match à venir dans la liste
3. Il appuie sur **Prédire**
4. Le backend récupère automatiquement les statistiques récentes des deux équipes, construit les variables (features), et interroge le modèle ML sélectionné comme le plus performant
5. Le résultat s'affiche sous forme de panneau type tableau de stade : probabilités, score probable, BTTS, Over/Under et facteurs explicatifs

Aucune intervention manuelle n'est requise après le clic sur Prédire.

## Données

Les données de matchs et de classements proviennent de l'API publique gratuite [football-data.org](https://www.football-data.org). Une clé API gratuite (tier "Free", 10 requêtes/minute) est nécessaire pour les données réelles.

**Sans clé API configurée, l'application fonctionne automatiquement en mode démonstration** avec des données simulées mais statistiquement réalistes, afin de rester utilisable immédiatement. Le mode actif (`live` ou `demo`) est indiqué dans chaque réponse de prédiction.

Statistiques utilisées pour la prédiction (31 variables) : force Elo, forme récente (points sur 5 matchs), xG/xGA, tirs, tirs cadrés, possession, corners, fautes, cartons, classement, blessures, jours de repos, série de victoires, taux de victoire en confrontations directes.

### Limites connues

- L'API gratuite football-data.org ne fournit pas directement xG, tirs ou possession détaillés : ces valeurs sont estimées à partir des résultats des matchs récents (proxy raisonnable, indiqué comme tel).
- Les données de blessures/suspensions/compositions probables ne sont pas disponibles sur le tier gratuit de l'API ; elles sont neutralisées (valeur par défaut) plutôt qu'inventées.
- Seules les grandes compétitions européennes sont couvertes par le tier gratuit (Premier League, Liga, Bundesliga, Serie A, Ligue 1, Champions League, Eredivisie, Primeira Liga).

## Modèles de Machine Learning

9 modèles ont été entraînés et comparés par validation croisée (5 folds) sur le résultat du match (Victoire domicile / Nul / Victoire extérieur) :

| Modèle | Précision (CV) | F1-macro |
|---|---|---|
| **Logistic Regression** ⭐ retenu | **58.9%** | 0.421 |
| Random Forest | 58.9% | 0.423 |
| Extra Trees | 58.4% | 0.417 |
| Gradient Boosting | 58.3% | 0.434 |
| KNN | 57.9% | 0.437 |
| XGBoost | 57.2% | 0.447 |
| LightGBM | 57.2% | 0.460 |
| SVM | 57.5% | 0.443 |
| Réseau de neurones (MLP) | 47.0% | 0.426 |

Le modèle le plus performant en validation croisée est sélectionné automatiquement (script `backend/app/ml/train_models.py`). Deux modèles auxiliaires (XGBoost) prédisent en complément BTTS (59.4% de précision) et Over/Under 2.5 buts (53.2%).

> À titre de repère, prédire l'issue exacte d'un match de football plafonne en réalité autour de 50-55% même pour les modèles professionnels, le football comportant une part d'aléa irréductible (forme du jour, décisions arbitrales, rebonds). Les probabilités affichées sont donc plus informatives que la prédiction binaire seule.

Le détail complet de la comparaison est disponible dans `backend/app/ml/model_comparison.json` et via l'endpoint `GET /api/model-info`.

### Ré-entraîner les modèles sur des données réelles

Le script `backend/app/ml/generate_training_data.py` génère actuellement un jeu de données synthétique statistiquement cohérent (corrélations réalistes entre force d'équipe, forme, xG et résultat) pour permettre un entraînement immédiat sans dépendance externe. Pour ré-entraîner sur des données historiques réelles :

1. Collecter l'historique des matchs terminés via `football_data_client.py` (déjà prêt à l'emploi avec une clé API)
2. Reproduire le format de colonnes de `training_data.csv`
3. Relancer `python app/ml/train_models.py`

## Démarrage local (développement)

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

## Déploiement

### Backend → Render
1. Créer un nouveau **Web Service** sur [render.com](https://render.com) à partir du dépôt GitHub, dossier `backend/`
2. Render détecte automatiquement `render.yaml` (build/run commands déjà configurées)
3. Ajouter la variable d'environnement `FOOTBALL_DATA_API_KEY` (optionnelle — mode démo sinon) dans Render → Environment
4. Noter l'URL publique générée (ex. `https://football-predictor-api.onrender.com`)

### Frontend → Vercel
1. Importer le dépôt GitHub sur [vercel.com](https://vercel.com), en sélectionnant le dossier `frontend/` comme racine du projet
2. Ajouter la variable d'environnement `NEXT_PUBLIC_API_URL` = URL du backend Render (étape précédente)
3. Déployer — Vercel détecte Next.js automatiquement

## Stack technique

- **Backend** : FastAPI, scikit-learn, XGBoost, LightGBM, httpx, Pydantic
- **Frontend** : Next.js 16 (App Router), React, TypeScript, Tailwind CSS 4
- **Données** : football-data.org (API publique gratuite)

## Sécurité & bonnes pratiques

- CORS configuré sur le backend pour accepter les appels du frontend
- Aucune clé API n'est exposée côté client (toutes les requêtes externes passent par le backend)
- Variables sensibles gérées exclusivement via variables d'environnement, jamais commitées
- Validation stricte des entrées via les schémas Pydantic
