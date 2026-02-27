#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import fisher_exact, pointbiserialr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "archive"
OUT = ROOT / "eda_workspace"
TABLES = OUT / "public_tables"
REPORT = OUT / "PUBLIC_EDA_REPORT.md"

TABLES.mkdir(parents=True, exist_ok=True)


def scan(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(path))


def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return np.nan


def hash_expr(col: str = "customer_id", pct: int = 10) -> pl.Expr:
    return (pl.col(col).hash(seed=SEED) % 100) < pct


def main() -> None:
    id_col = "customer_id"

    train_main_schema = scan(ARCH / "train_main_features.parquet").collect_schema().names()
    train_extra_schema = scan(ARCH / "train_extra_features.parquet").collect_schema().names()
    target_cols = [c for c in scan(ARCH / "train_target.parquet").collect_schema().names() if c != id_col]

    num_main = [c for c in train_main_schema if c.startswith("num_feature_")]
    cat_main = [c for c in train_main_schema if c.startswith("cat_feature_")]
    extra_cols = [c for c in train_extra_schema if c != id_col]
    all_features = [c for c in train_main_schema if c != id_col] + extra_cols

    n_train = int(scan(ARCH / "train_main_features.parquet").select(pl.len()).collect(engine="streaming").item())
    n_test = int(scan(ARCH / "test_main_features.parquet").select(pl.len()).collect(engine="streaming").item())

    # Target stats
    target_stats = (
        scan(ARCH / "train_target.parquet")
        .select([
            pl.col(t).cast(pl.Int64).sum().alias(f"{t}__pos") for t in target_cols
        ])
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    rows = []
    for t in target_cols:
        pos = int(target_stats[f"{t}__pos"])
        rows.append({"target": t, "positive_count": pos, "positive_rate": pos / n_train})
    target_df = pd.DataFrame(rows).sort_values("positive_rate", ascending=False)
    target_df.to_csv(TABLES / "target_stats.csv", index=False)

    # Correlation + clustering
    y_df = (
        scan(ARCH / "train_target.parquet")
        .select([pl.col(t).cast(pl.Int8).alias(t) for t in target_cols])
        .collect(engine="streaming")
        .to_pandas()
    )
    corr = y_df.corr(method="pearson")
    corr.to_csv(TABLES / "target_correlation_matrix.csv")

    corr_t10 = corr.loc["target_10_1"].drop("target_10_1")
    t10_profile = pd.DataFrame({
        "other_target": corr_t10.index,
        "correlation": corr_t10.values,
        "abs_correlation": np.abs(corr_t10.values),
    }).sort_values("abs_correlation", ascending=False)
    t10_profile.to_csv(TABLES / "target_10_1_profile.csv", index=False)

    dist = 1.0 - np.abs(corr.to_numpy(dtype=float))
    cl_rows = []
    for k in [3, 4, 5]:
        try:
            cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        except TypeError:
            cl = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
        labels = cl.fit_predict(dist)
        sil = float(silhouette_score(dist, labels, metric="precomputed")) if len(np.unique(labels)) > 1 else np.nan
        counts = pd.Series(labels).value_counts()
        cl_rows.append({
            "k": k,
            "silhouette_precomputed": sil,
            "largest_cluster_share": float(counts.max() / len(target_cols)),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        })
    cl_eval = pd.DataFrame(cl_rows)
    cl_eval.to_csv(TABLES / "target_cluster_quality.csv", index=False)

    # Missingness summary top10 (extra only)
    miss_lf = scan(ARCH / "train_extra_features.parquet").select([
        pl.col(c).is_null().mean().alias(c) for c in extra_cols
    ])
    miss = miss_lf.collect(engine="streaming").to_dicts()[0]
    miss_df = pd.DataFrame({"feature": list(miss.keys()), "null_rate": list(miss.values())}).sort_values("null_rate", ascending=False)
    miss_df.head(10).to_csv(TABLES / "top10_missing_features.csv", index=False)

    # Missingness as signal (filled count)
    open_cols = [c for c in target_cols if c != "target_10_1"]
    opened_expr = pl.sum_horizontal([pl.col(c).cast(pl.Int16) for c in open_cols]).alias("opened_non10_1")
    fill_df = (
        scan(ARCH / "train_extra_features.parquet")
        .select([
            pl.col(id_col),
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int16) for c in extra_cols]).alias("filled_extra_count"),
        ])
        .join(
            scan(ARCH / "train_target.parquet").select([pl.col(id_col), opened_expr]),
            on=id_col,
            how="inner",
        )
        .with_columns((pl.col("opened_non10_1") > 0).cast(pl.Int8).alias("target_any_open_non10_1"))
        .collect(engine="streaming")
        .to_pandas()
    )
    y_any = fill_df["target_any_open_non10_1"].to_numpy(dtype=np.int8)
    x_fill = fill_df["filled_extra_count"].to_numpy(dtype=float)
    pb_corr, pb_p = pointbiserialr(y_any, x_fill)
    auc_fill = safe_auc(y_any, x_fill)

    # Cardinality + unseen categories
    card_rows = []
    unseen_rows = []
    for c in cat_main:
        tr_uni = int(scan(ARCH / "train_main_features.parquet").select(pl.col(c).drop_nulls().n_unique().alias("n")).collect(engine="streaming")["n"][0])
        te_uni = int(scan(ARCH / "test_main_features.parquet").select(pl.col(c).drop_nulls().n_unique().alias("n")).collect(engine="streaming")["n"][0])

        tr_set = set(scan(ARCH / "train_main_features.parquet").select(pl.col(c).drop_nulls().unique()).collect(engine="streaming")[c].to_list())
        te_vals = scan(ARCH / "test_main_features.parquet").select(pl.col(c).drop_nulls()).collect(engine="streaming")[c].to_list()

        unseen_rate = float(np.mean([v not in tr_set for v in te_vals])) if te_vals else 0.0
        unseen_unique = int(sum(v not in tr_set for v in set(te_vals))) if te_vals else 0

        card_rows.append({"feature": c, "train_nunique": tr_uni, "test_nunique": te_uni})
        unseen_rows.append({"feature": c, "unseen_unique_categories": unseen_unique, "unseen_rate_test_rows": unseen_rate})

    card_df = pd.DataFrame(card_rows).sort_values("train_nunique", ascending=False)
    unseen_df = pd.DataFrame(unseen_rows).sort_values("unseen_rate_test_rows", ascending=False)
    card_df.to_csv(TABLES / "categorical_cardinality.csv", index=False)
    unseen_df.to_csv(TABLES / "categorical_unseen_categories.csv", index=False)

    # Adversarial shift (main features only, 20% sample)
    adv_num = num_main
    adv_cat = cat_main
    main_expr = [pl.col(c).fill_null(-1).cast(pl.Int16).alias(c) if c in adv_cat else pl.col(c).cast(pl.Float32).alias(c) for c in (adv_num + adv_cat)]

    adv_train = (
        scan(ARCH / "train_main_features.parquet")
        .filter(hash_expr(id_col, 20))
        .select([pl.col(id_col)] + main_expr)
        .collect(engine="streaming")
        .to_pandas()
    )
    adv_test = (
        scan(ARCH / "test_main_features.parquet")
        .filter(hash_expr(id_col, 20))
        .select([pl.col(id_col)] + main_expr)
        .collect(engine="streaming")
        .to_pandas()
    )
    feat_adv = adv_num + adv_cat
    X_adv = pd.concat([adv_train[feat_adv], adv_test[feat_adv]], axis=0, ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(adv_train), dtype=np.int8), np.ones(len(adv_test), dtype=np.int8)])
    X_tr, X_va, y_tr, y_va = train_test_split(X_adv, y_adv, test_size=0.25, stratify=y_adv, random_state=SEED)
    cat_idx = [i for i, c in enumerate(feat_adv) if c in adv_cat]

    adv_model = CatBoostClassifier(
        iterations=120,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )
    adv_model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
    adv_auc = safe_auc(y_va, adv_model.predict_proba(X_va)[:, 1])

    # Outlier whales (main numeric, rare targets)
    rare_targets = target_df[target_df["positive_rate"] < 0.005]["target"].tolist()
    whale_feats = num_main[:]

    o_df = (
        scan(ARCH / "train_main_features.parquet")
        .filter(hash_expr(id_col, 12))
        .select([pl.col(id_col)] + [pl.col(f).cast(pl.Float32) for f in whale_feats])
        .join(
            scan(ARCH / "train_target.parquet")
            .filter(hash_expr(id_col, 12))
            .select([pl.col(id_col)] + [pl.col(t).cast(pl.Int8) for t in rare_targets]),
            on=id_col,
            how="inner",
        )
        .collect(engine="streaming")
        .to_pandas()
    )

    whale_rows = []
    for f in whale_feats:
        x = o_df[f].to_numpy(dtype=float)
        if np.all(np.isnan(x)):
            continue
        q99 = np.nanquantile(x, 0.99)
        top = x >= q99
        rest = ~top
        if top.sum() < 50 or rest.sum() < 1000:
            continue
        for t in rare_targets:
            y = o_df[t].to_numpy(dtype=np.int8)
            tp, tn = int((y[top] == 1).sum()), int(top.sum())
            rp, rn = int((y[rest] == 1).sum()), int(rest.sum())
            if rp == 0:
                continue
            lift = (tp / tn) / (rp / rn)
            _, p = fisher_exact([[tp, tn - tp], [rp, rn - rp]], alternative="greater")
            whale_rows.append({"target": t, "feature": f, "lift": float(lift), "pvalue": float(p)})

    whale_df = pd.DataFrame(whale_rows)
    whale_sig = whale_df[(whale_df["lift"] >= 2.0) & (whale_df["pvalue"] < 0.05)].sort_values("lift", ascending=False)
    whale_sig.to_csv(TABLES / "whale_signals.csv", index=False)
    whale_summary = (
        whale_sig.groupby("feature", as_index=False)
        .agg(n_rare_targets=("target", "nunique"), median_lift=("lift", "median"), max_lift=("lift", "max"))
        .sort_values(["n_rare_targets", "median_lift"], ascending=[False, False])
    )
    whale_summary.to_csv(TABLES / "whale_feature_candidates.csv", index=False)

    # Golden linear features for selected targets on 8% sample
    sel_targets = ["target_1_1", "target_3_2", "target_10_1", "target_9_6"]
    lin_feats = [c for c in all_features if c.startswith("cat_feature_")] + [c for c in all_features if c.startswith("num_feature_")][:400]
    main_in = [c for c in lin_feats if c in train_main_schema]
    extra_in = [c for c in lin_feats if c in train_extra_schema]

    l_df = (
        scan(ARCH / "train_main_features.parquet")
        .filter(hash_expr(id_col, 8))
        .select([pl.col(id_col)] + [pl.col(c).cast(pl.Float32).alias(c) for c in main_in])
        .join(
            scan(ARCH / "train_extra_features.parquet")
            .filter(hash_expr(id_col, 8))
            .select([pl.col(id_col)] + [pl.col(c).cast(pl.Float32).alias(c) for c in extra_in]),
            on=id_col,
            how="inner",
        )
        .join(
            scan(ARCH / "train_target.parquet")
            .filter(hash_expr(id_col, 8))
            .select([pl.col(id_col)] + [pl.col(t).cast(pl.Int8) for t in sel_targets]),
            on=id_col,
            how="inner",
        )
        .collect(engine="streaming")
        .to_pandas()
    )

    feat_lin = main_in + extra_in
    X = l_df[feat_lin].to_numpy(dtype=np.float32).copy()
    means = np.nanmean(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(means, idx[1])
    X = X - X.mean(axis=0)
    stdx = X.std(axis=0)
    stdx[stdx < 1e-12] = np.nan

    lin_rows = []
    for t in sel_targets:
        y = l_df[t].to_numpy(dtype=np.float32)
        y = y - y.mean()
        sdy = y.std()
        if sdy < 1e-12:
            continue
        cov = (X.T @ y) / len(y)
        cvec = cov / (stdx * sdy)
        for f, c in zip(feat_lin, cvec):
            if np.isfinite(c):
                lin_rows.append({"target": t, "feature": f, "pearson_corr": float(c), "abs_corr": float(abs(c))})
    lin_df = pd.DataFrame(lin_rows).sort_values(["target", "abs_corr"], ascending=[True, False])
    lin_top = lin_df.groupby("target", as_index=False).head(5)
    lin_top.to_csv(TABLES / "golden_linear_top5_selected_targets.csv", index=False)

    # Report
    n_lt_1 = int((target_df["positive_rate"] < 0.01).sum())
    n_lt_50 = int((target_df["positive_count"] < 50).sum())
    min_pos = int(target_df["positive_count"].min())
    neg_share_t10 = float((corr_t10 < 0).mean())

    clear_4 = bool(
        (float(cl_eval.loc[cl_eval["k"] == 4, "largest_cluster_share"].iloc[0]) <= 0.60)
        and (float(cl_eval.loc[cl_eval["k"] == 4, "silhouette_precomputed"].iloc[0]) >= 0.08)
    )

    n_unseen_feats = int((unseen_df["unseen_unique_categories"] > 0).sum())

    summary = {
        "rows_train": n_train,
        "rows_test": n_test,
        "n_targets": len(target_cols),
        "n_features_main": len(train_main_schema) - 1,
        "n_features_extra": len(extra_cols),
        "targets_lt_1pct": n_lt_1,
        "targets_lt_50": n_lt_50,
        "min_positive_count": min_pos,
        "target_10_1_negative_share": neg_share_t10,
        "filled_extra_count_auc": float(auc_fill),
        "filled_extra_count_pointbiserial": float(pb_corr),
        "adversarial_auc_main_features": float(adv_auc),
        "cat_features_with_unseen_in_test": n_unseen_feats,
        "clear_4_target_clusters": clear_4,
        "significant_whale_pairs": int(len(whale_sig)),
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rep = f"""# Public EDA Report: Multi-Label Banking Products (41 targets)

## 1. Data Landscape
Исследование проведено на train/test parquet без загрузки полного датасета в pandas: агрегаты и выборки считались через Polars (lazy/streaming).

- Train rows: **{n_train}**
- Test rows: **{n_test}**
- Targets: **{len(target_cols)}**
- Main features: **{len(train_main_schema) - 1}**
- Extra features: **{len(extra_cols)}**

## 2. Target Structure and Imbalance
- Targets with prevalence <1%: **{n_lt_1} / {len(target_cols)}**
- Targets with fewer than 50 positives: **{n_lt_50}**
- Minimum positive count among targets: **{min_pos}**

Вывод: задача существенно несбалансированная, но без экстремума "пара десятков наблюдений". Нужна стабильная per-target валидация и контроль разброса AUC по фолдам.

## 3. Inter-Target Dependencies
`target_10_1` показывает антагонистичную роль:
- Share negative correlations vs other targets: **{neg_share_t10:.2%}**

Кластеризация таргетов по |corr|-distance не дала чистого разбиения на 4 устойчивых группы:
- clear 4 clusters: **{'yes' if clear_4 else 'no'}**
- детали: `public_tables/target_cluster_quality.csv`

Вывод: 41 отдельных модели остаются базовой стратегией; межтаргетные связи лучше добавлять через OOF meta-features, а не через жесткую замену на 4 мета-модели.

## 4. Missingness Signal
Проверен глобальный сигнал заполненности extra-фич:
- AUC(`filled_extra_count` -> any open except `target_10_1`): **{auc_fill:.4f}**
- Point-biserial correlation: **{pb_corr:.4f}** (p={pb_p:.2e})

Вывод: наличие заполненных extra-фич связано с активностью клиента и должно использоваться как источник признаков (`filled_extra_count`, null-indicators).

## 5. Train/Test Shift and Categorical Safety
Adversarial test (train vs test, main features):
- AUC: **{adv_auc:.4f}**

Новые категории в test:
- cat-features with unseen categories: **{n_unseen_feats}**
- детали: `public_tables/categorical_unseen_categories.csv`

Вывод: глобального сильного covariate shift не видно, но categorical encoder обязан поддерживать unknown значения (`handle_unknown='value'`).

## 6. Outliers and Whale Behaviour
Для редких таргетов проверен lift top-1% по числовым main-фичам.
- Significant whale pairs (lift>=2, p<0.05): **{len(whale_sig)}**
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
"""
    REPORT.write_text(rep, encoding="utf-8")
    print("public pipeline done")


if __name__ == "__main__":
    main()
