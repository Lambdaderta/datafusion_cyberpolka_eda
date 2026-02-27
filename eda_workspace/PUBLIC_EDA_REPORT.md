# Public EDA Report: Multi-Label Banking Products (41 targets)

## 1. Data Landscape
Исследование проведено на train/test parquet без загрузки полного датасета в pandas: агрегаты и выборки считались через Polars (lazy/streaming).

- Train rows: **750000**
- Test rows: **250000**
- Targets: **41**
- Main features: **199**
- Extra features: **2241**

## 2. Target Structure and Imbalance
- Targets with prevalence <1%: **24 / 41**
- Targets with fewer than 50 positives: **0**
- Minimum positive count among targets: **83**

Вывод: задача существенно несбалансированная, но без экстремума "пара десятков наблюдений". Нужна стабильная per-target валидация и контроль разброса AUC по фолдам.

## 3. Inter-Target Dependencies
`target_10_1` показывает антагонистичную роль:
- Share negative correlations vs other targets: **100.00%**

Кластеризация таргетов по |corr|-distance не дала чистого разбиения на 4 устойчивых группы:
- clear 4 clusters: **no**
- детали: `public_tables/target_cluster_quality.csv`

Вывод: 41 отдельных модели остаются базовой стратегией; межтаргетные связи лучше добавлять через OOF meta-features, а не через жесткую замену на 4 мета-модели.

## 4. Missingness Signal
Проверен глобальный сигнал заполненности extra-фич:
- AUC(`filled_extra_count` -> any open except `target_10_1`): **0.6520**
- Point-biserial correlation: **0.2509** (p=0.00e+00)

Вывод: наличие заполненных extra-фич связано с активностью клиента и должно использоваться как источник признаков (`filled_extra_count`, null-indicators).

## 5. Train/Test Shift and Categorical Safety
Adversarial test (train vs test, main features):
- AUC: **0.5007**

Новые категории в test:
- cat-features with unseen categories: **2**
- детали: `public_tables/categorical_unseen_categories.csv`

Вывод: глобального сильного covariate shift не видно, но categorical encoder обязан поддерживать unknown значения (`handle_unknown='value'`).

## 6. Outliers and Whale Behaviour
Для редких таргетов проверен lift top-1% по числовым main-фичам.
- Significant whale pairs (lift>=2, p<0.05): **313**
- shortlist features: `public_tables/whale_feature_candidates.csv`

Вывод: признаки `is_whale` по shortlist дают практический потенциал буста для редких продуктов.

## 7. Golden Features (Linear Screen)
Для `target_1_1`, `target_3_2`, `target_10_1`, `target_9_6` рассчитаны top-5 линейных сигналов.
- таблица: `public_tables/golden_linear_top5_selected_targets.csv`

## 8. Practical Modeling Guidelines
- Стартовая схема: независимые 41 модели + OOF meta-features по коррелированным таргетам.
- Для encoder-ов категорий: обязательно unknown handling и устойчивый prior.
- Для редких таргетов: усилить регуляризацию, отслеживать fold-variance, добавлять whale/null признаки.

## Artifacts
- Pipeline code: `eda_workspace/public_eda_pipeline.py`
- This report: `eda_workspace/PUBLIC_EDA_REPORT.md`
- Core tables: `eda_workspace/public_tables/`
