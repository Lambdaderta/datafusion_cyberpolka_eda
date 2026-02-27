# Public EDA Pipeline (Multi-Label, 41 targets)

Минимальная публичная версия репозитория:
- один воспроизводимый pipeline;
- один полный структурированный отчет;
- компактные таблицы с ключевыми выводами.

## Структура
- `eda_workspace/public_eda_pipeline.py` — единый EDA pipeline
- `eda_workspace/PUBLIC_EDA_REPORT.md` — итоговый отчет
- `eda_workspace/public_tables/` — ключевые артефакты (CSV/JSON)
- `archive/*.parquet` — исходные данные

## Запуск
```bash
.venv/bin/python eda_workspace/public_eda_pipeline.py
```

## Что исследовано в отчете
- структура данных и дисбаланс таргетов;
- зависимости между таргетами и проверка кластеризации;
- сигнал в пропусках;
- train/test shift (adversarial AUC);
- unseen категории в test;
- outlier/whale эффект для редких таргетов;
- широкий feature-target скрининг по всем 41 таргетам;
- golden features (линейный скрининг + universality фич);
- практические рекомендации для обучения моделей.

## Быстрый индекс таблиц
- `public_tables/target_top10_features_linear.csv` — топ-10 линейных фич для каждого таргета (41 таргет)
- `public_tables/target_top10_feature_mix.csv` — профиль таргета: cat/num и main/extra в топ-сигналах
- `public_tables/feature_universality_top10.csv` — универсальные фичи (сколько таргетов покрывают)
- `public_tables/feature_signal_summary.csv` — глобальная сила фич по всем таргетам
- `public_tables/target_pair_stats.csv` — пары таргетов: корреляции, co-occurrence, lift
- `public_tables/top_positive_target_pairs.csv` / `top_negative_target_pairs.csv` — самые сильные связи
- `public_tables/whale_top3_per_target.csv` — top whale-взаимодействия по каждому редкому таргету
- `public_tables/missing_indicator_auc_popular_targets.csv` — сигнал индикаторов пропусков
