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
- golden features (линейный скрининг для ключевых таргетов);
- практические рекомендации для обучения моделей.
