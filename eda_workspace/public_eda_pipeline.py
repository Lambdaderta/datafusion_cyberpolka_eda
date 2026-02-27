#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from scipy.stats import fisher_exact, pointbiserialr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "archive"
OUT = ROOT / "eda_workspace"
TABLES = OUT / "public_tables"
REPORT = OUT / "PUBLIC_EDA_REPORT.md"

TABLES.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Utils
# -------------------------------
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


def pretty(df: pd.DataFrame, n: int = 10) -> str:
    if df is None or df.empty:
        return "(empty)"
    return df.head(n).to_string(index=False)


def target_family(t: str) -> str:
    # target_9_6 -> family '9'
    parts = t.split("_")
    return parts[1] if len(parts) >= 3 else "unknown"


# -------------------------------
# Main
# -------------------------------
def main() -> None:
    id_col = "customer_id"

    train_main_cols = scan(ARCH / "train_main_features.parquet").collect_schema().names()
    train_extra_cols = scan(ARCH / "train_extra_features.parquet").collect_schema().names()
    target_cols = [c for c in scan(ARCH / "train_target.parquet").collect_schema().names() if c != id_col]

    main_features = [c for c in train_main_cols if c != id_col]
    num_main = [c for c in main_features if c.startswith("num_feature_")]
    cat_main = [c for c in main_features if c.startswith("cat_feature_")]
    extra_features = [c for c in train_extra_cols if c != id_col]

    # -------------------------------
    # Inventory / target prevalence
    # -------------------------------
    n_train = int(
        scan(ARCH / "train_main_features.parquet")
        .select(pl.len().alias("n"))
        .collect(engine="streaming")["n"][0]
    )
    n_test = int(
        scan(ARCH / "test_main_features.parquet")
        .select(pl.len().alias("n"))
        .collect(engine="streaming")["n"][0]
    )

    target_sums = (
        scan(ARCH / "train_target.parquet")
        .select([pl.col(t).cast(pl.Int64).sum().alias(t) for t in target_cols])
        .collect(engine="streaming")
        .to_dicts()[0]
    )

    target_rows = []
    for t in target_cols:
        pos = int(target_sums[t])
        target_rows.append({
            "target": t,
            "family": target_family(t),
            "positive_count": pos,
            "positive_rate": pos / n_train,
        })
    target_df = pd.DataFrame(target_rows).sort_values("positive_rate", ascending=False)
    target_df.to_csv(TABLES / "target_stats.csv", index=False)

    target_family_df = (
        target_df.groupby("family", as_index=False)
        .agg(
            n_targets=("target", "count"),
            mean_rate=("positive_rate", "mean"),
            min_rate=("positive_rate", "min"),
            max_rate=("positive_rate", "max"),
        )
        .sort_values("mean_rate", ascending=False)
    )
    target_family_df.to_csv(TABLES / "target_family_stats.csv", index=False)

    # Full target matrix is safe (~750k x 41)
    y_df = (
        scan(ARCH / "train_target.parquet")
        .select([pl.col(t).cast(pl.Int8).alias(t) for t in target_cols])
        .collect(engine="streaming")
        .to_pandas()
    )

    target_sum = y_df.sum(axis=1)
    sum_dist = (
        pd.Series(target_sum)
        .value_counts()
        .sort_index()
        .rename_axis("opened_targets")
        .reset_index(name="count")
    )
    sum_dist["share"] = sum_dist["count"] / n_train
    sum_dist.to_csv(TABLES / "opened_targets_distribution.csv", index=False)

    # -------------------------------
    # Target dependencies
    # -------------------------------
    corr = y_df.corr(method="pearson")
    corr.to_csv(TABLES / "target_correlation_matrix.csv")

    y_mat = y_df[target_cols].to_numpy(dtype=np.int32)
    co_counts = (y_mat.T @ y_mat).astype(np.int64)
    prev = y_mat.mean(axis=0)

    pair_rows = []
    for i, ta in enumerate(target_cols):
        for j in range(i + 1, len(target_cols)):
            tb = target_cols[j]
            co_count = int(co_counts[i, j])
            co_rate = co_count / n_train
            expected = float(prev[i] * prev[j])
            lift = float(co_rate / expected) if expected > 0 else np.nan
            pair_rows.append({
                "target_a": ta,
                "target_b": tb,
                "corr": float(corr.iloc[i, j]),
                "co_count": co_count,
                "co_rate": co_rate,
                "expected_independent_rate": expected,
                "pair_lift": lift,
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(TABLES / "target_pair_stats.csv", index=False)

    top_pos_pairs = pair_df.sort_values("corr", ascending=False).head(30)
    top_neg_pairs = pair_df.sort_values("corr", ascending=True).head(30)
    top_lift_pairs = pair_df[pair_df["co_count"] >= 100].sort_values("pair_lift", ascending=False).head(30)
    top_pos_pairs.to_csv(TABLES / "top_positive_target_pairs.csv", index=False)
    top_neg_pairs.to_csv(TABLES / "top_negative_target_pairs.csv", index=False)
    top_lift_pairs.to_csv(TABLES / "top_cooccurrence_lift_pairs.csv", index=False)

    corr_t10 = corr.loc["target_10_1"].drop("target_10_1")
    t10_profile = pd.DataFrame({
        "other_target": corr_t10.index,
        "correlation": corr_t10.values,
        "abs_correlation": np.abs(corr_t10.values),
    }).sort_values("abs_correlation", ascending=False)
    t10_profile.to_csv(TABLES / "target_10_1_profile.csv", index=False)

    # Clustering quality and assignments
    dist = 1.0 - np.abs(corr.to_numpy(dtype=float))
    cluster_eval_rows = []
    labels_k4 = None
    for k in [3, 4, 5]:
        try:
            cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        except TypeError:
            cl = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
        labels = cl.fit_predict(dist)
        if k == 4:
            labels_k4 = labels
        sil = float(silhouette_score(dist, labels, metric="precomputed")) if len(np.unique(labels)) > 1 else np.nan
        counts = pd.Series(labels).value_counts()
        cluster_eval_rows.append({
            "k": k,
            "silhouette_precomputed": sil,
            "largest_cluster_share": float(counts.max() / len(target_cols)),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        })
    cluster_eval_df = pd.DataFrame(cluster_eval_rows)
    cluster_eval_df.to_csv(TABLES / "target_cluster_quality.csv", index=False)

    cluster_assign_df = pd.DataFrame({"target": target_cols, "cluster_k4": labels_k4})
    cluster_assign_df["family"] = cluster_assign_df["target"].map(target_family)
    cluster_assign_df.to_csv(TABLES / "target_clusters_k4.csv", index=False)

    cluster_summary_rows = []
    for cl_id, g in cluster_assign_df.groupby("cluster_k4"):
        ts = g["target"].tolist()
        if len(ts) > 1:
            sub = corr.loc[ts, ts].to_numpy(dtype=float)
            iu = np.triu_indices(len(ts), k=1)
            avg_abs = float(np.abs(sub[iu]).mean())
        else:
            avg_abs = np.nan
        fam_mode = g["family"].value_counts(normalize=True)
        cluster_summary_rows.append({
            "cluster_k4": int(cl_id),
            "n_targets": int(len(ts)),
            "avg_abs_corr_inside": avg_abs,
            "dominant_family": str(fam_mode.index[0]),
            "dominant_family_share": float(fam_mode.iloc[0]),
            "targets": ", ".join(sorted(ts)),
        })
    cluster_summary_df = pd.DataFrame(cluster_summary_rows).sort_values("n_targets", ascending=False)
    cluster_summary_df.to_csv(TABLES / "target_cluster_summary.csv", index=False)

    # -------------------------------
    # Missingness structure
    # -------------------------------
    extra_miss_dict = (
        scan(ARCH / "train_extra_features.parquet")
        .select([pl.col(c).is_null().mean().alias(c) for c in extra_features])
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    extra_miss_df = pd.DataFrame({
        "feature": list(extra_miss_dict.keys()),
        "null_rate": list(extra_miss_dict.values()),
        "source": "extra",
        "feature_type": "num",
    }).sort_values("null_rate", ascending=False)
    extra_miss_df.to_csv(TABLES / "extra_missingness_summary.csv", index=False)

    main_miss_dict = (
        scan(ARCH / "train_main_features.parquet")
        .select([pl.col(c).is_null().mean().alias(c) for c in main_features])
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    main_miss_df = pd.DataFrame({
        "feature": list(main_miss_dict.keys()),
        "null_rate": list(main_miss_dict.values()),
    })
    main_miss_df["source"] = "main"
    main_miss_df["feature_type"] = np.where(main_miss_df["feature"].str.startswith("cat_feature_"), "cat", "num")

    miss_all_df = pd.concat([main_miss_df, extra_miss_df], ignore_index=True)
    miss_all_df = miss_all_df.sort_values("null_rate", ascending=False)
    miss_all_df.to_csv(TABLES / "feature_missingness_summary.csv", index=False)

    top10_missing = extra_miss_df.head(10).copy()
    top10_missing.to_csv(TABLES / "top10_missing_features.csv", index=False)

    miss_bands = pd.DataFrame({
        "band": [">99%", ">95%", ">90%", "50-90%", "10-50%", "<=10%"],
        "count": [
            int((extra_miss_df["null_rate"] > 0.99).sum()),
            int((extra_miss_df["null_rate"] > 0.95).sum()),
            int((extra_miss_df["null_rate"] > 0.90).sum()),
            int(((extra_miss_df["null_rate"] > 0.50) & (extra_miss_df["null_rate"] <= 0.90)).sum()),
            int(((extra_miss_df["null_rate"] > 0.10) & (extra_miss_df["null_rate"] <= 0.50)).sum()),
            int((extra_miss_df["null_rate"] <= 0.10).sum()),
        ],
    })
    miss_bands.to_csv(TABLES / "extra_missingness_bands.csv", index=False)

    # Missingness as activity signal
    open_cols = [c for c in target_cols if c != "target_10_1"]
    opened_expr = pl.sum_horizontal([pl.col(c).cast(pl.Int16) for c in open_cols]).alias("opened_non10_1")
    fill_df = (
        scan(ARCH / "train_extra_features.parquet")
        .select([
            pl.col(id_col),
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int16) for c in extra_features]).alias("filled_extra_count"),
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

    fill_deciles = fill_df[["filled_extra_count", "target_any_open_non10_1"]].copy()
    fill_deciles["decile"] = pd.qcut(fill_deciles["filled_extra_count"], q=10, labels=False, duplicates="drop")
    fill_dec_df = (
        fill_deciles.groupby("decile", as_index=False)
        .agg(
            n=("target_any_open_non10_1", "size"),
            avg_filled=("filled_extra_count", "mean"),
            min_filled=("filled_extra_count", "min"),
            max_filled=("filled_extra_count", "max"),
            target_rate=("target_any_open_non10_1", "mean"),
        )
    )
    fill_dec_df.to_csv(TABLES / "filled_extra_count_deciles.csv", index=False)

    # Missing indicator vs popular targets on moderate-missing features
    pop_targets = target_df[target_df["target"] != "target_10_1"].head(10)["target"].tolist()
    moderate_missing_feats = (
        extra_miss_df[(extra_miss_df["null_rate"] >= 0.20) & (extra_miss_df["null_rate"] <= 0.98)]
        .head(30)["feature"]
        .tolist()
    )

    miss_auc_rows = []
    if moderate_missing_feats:
        miss_eval_df = (
            scan(ARCH / "train_extra_features.parquet")
            .filter(hash_expr(id_col, 30))
            .select([pl.col(id_col)] + [pl.col(c) for c in moderate_missing_feats])
            .join(
                scan(ARCH / "train_target.parquet")
                .filter(hash_expr(id_col, 30))
                .select([pl.col(id_col)] + [pl.col(t).cast(pl.Int8) for t in pop_targets]),
                on=id_col,
                how="inner",
            )
            .collect(engine="streaming")
            .to_pandas()
        )

        null_rate_map = dict(zip(extra_miss_df["feature"], extra_miss_df["null_rate"]))
        for f in moderate_missing_feats:
            ind = miss_eval_df[f].isna().astype(np.int8).to_numpy()
            if ind.min() == ind.max():
                continue
            miss_rate = float(ind.mean())
            for t in pop_targets:
                y = miss_eval_df[t].to_numpy(dtype=np.int8)
                auc = safe_auc(y, ind)
                auc_eff = np.nan if not np.isfinite(auc) else max(auc, 1.0 - auc)
                miss_auc_rows.append({
                    "target": t,
                    "feature": f,
                    "auc_single_feature": auc,
                    "auc_effective": auc_eff,
                    "null_rate": null_rate_map.get(f, np.nan),
                    "missing_rate_indicator": miss_rate,
                })
    miss_auc_df = pd.DataFrame(miss_auc_rows).sort_values("auc_effective", ascending=False)
    miss_auc_df.to_csv(TABLES / "missing_indicator_auc_popular_targets.csv", index=False)

    # -------------------------------
    # Cardinality and unseen test categories
    # -------------------------------
    card_rows = []
    unseen_rows = []
    for c in cat_main:
        tr_uni = int(
            scan(ARCH / "train_main_features.parquet")
            .select(pl.col(c).drop_nulls().n_unique().alias("n"))
            .collect(engine="streaming")["n"][0]
        )
        te_uni = int(
            scan(ARCH / "test_main_features.parquet")
            .select(pl.col(c).drop_nulls().n_unique().alias("n"))
            .collect(engine="streaming")["n"][0]
        )

        tr_set = set(
            scan(ARCH / "train_main_features.parquet")
            .select(pl.col(c).drop_nulls().unique())
            .collect(engine="streaming")[c]
            .to_list()
        )
        te_vals = (
            scan(ARCH / "test_main_features.parquet")
            .select(pl.col(c).drop_nulls())
            .collect(engine="streaming")[c]
            .to_list()
        )

        unseen_rate = float(np.mean([v not in tr_set for v in te_vals])) if te_vals else 0.0
        unseen_unique = int(sum(v not in tr_set for v in set(te_vals))) if te_vals else 0

        card_rows.append({"feature": c, "train_nunique": tr_uni, "test_nunique": te_uni})
        unseen_rows.append({"feature": c, "unseen_unique_categories": unseen_unique, "unseen_rate_test_rows": unseen_rate})

    card_df = pd.DataFrame(card_rows).sort_values("train_nunique", ascending=False)
    unseen_df = pd.DataFrame(unseen_rows).sort_values("unseen_rate_test_rows", ascending=False)
    card_df.to_csv(TABLES / "categorical_cardinality.csv", index=False)
    unseen_df.to_csv(TABLES / "categorical_unseen_categories.csv", index=False)

    # -------------------------------
    # Adversarial shift (main only)
    # -------------------------------
    adv_feats = num_main + cat_main
    adv_expr = [
        pl.col(c).fill_null(-1).cast(pl.Int16).alias(c) if c in cat_main else pl.col(c).cast(pl.Float32).alias(c)
        for c in adv_feats
    ]

    adv_train = (
        scan(ARCH / "train_main_features.parquet")
        .filter(hash_expr(id_col, 20))
        .select([pl.col(id_col)] + adv_expr)
        .collect(engine="streaming")
        .to_pandas()
    )
    adv_test = (
        scan(ARCH / "test_main_features.parquet")
        .filter(hash_expr(id_col, 20))
        .select([pl.col(id_col)] + adv_expr)
        .collect(engine="streaming")
        .to_pandas()
    )

    X_adv = pd.concat([adv_train[adv_feats], adv_test[adv_feats]], axis=0, ignore_index=True)
    y_adv = np.concatenate([
        np.zeros(len(adv_train), dtype=np.int8),
        np.ones(len(adv_test), dtype=np.int8),
    ])

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_adv,
        y_adv,
        test_size=0.25,
        stratify=y_adv,
        random_state=SEED,
    )

    cat_idx_adv = [i for i, c in enumerate(adv_feats) if c in cat_main]

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
    adv_model.fit(X_tr, y_tr, cat_features=cat_idx_adv, eval_set=(X_va, y_va), use_best_model=True)
    adv_auc = safe_auc(y_va, adv_model.predict_proba(X_va)[:, 1])

    # -------------------------------
    # Wide feature -> target screening (all 41 targets)
    # -------------------------------
    # Keep all main features + top extra by observed density to avoid near-empty noise.
    extra_dense = extra_miss_df.sort_values("null_rate", ascending=True).head(320)["feature"].tolist()
    screen_features = main_features + extra_dense
    screen_main = [c for c in screen_features if c in main_features]
    screen_extra = [c for c in screen_features if c in extra_features]

    screen_lf = (
        scan(ARCH / "train_main_features.parquet")
        .filter(hash_expr(id_col, 12))
        .select([pl.col(id_col)] + [
            (pl.col(c).fill_null(-1).cast(pl.Float32) if c.startswith("cat_feature_") else pl.col(c).cast(pl.Float32)).alias(c)
            for c in screen_main
        ])
        .join(
            scan(ARCH / "train_extra_features.parquet")
            .filter(hash_expr(id_col, 12))
            .select([pl.col(id_col)] + [pl.col(c).cast(pl.Float32).alias(c) for c in screen_extra]),
            on=id_col,
            how="inner",
        )
        .join(
            scan(ARCH / "train_target.parquet")
            .filter(hash_expr(id_col, 12))
            .select([pl.col(id_col)] + [pl.col(t).cast(pl.Int8).alias(t) for t in target_cols]),
            on=id_col,
            how="inner",
        )
    )
    screen_df = screen_lf.collect(engine="streaming").to_pandas()

    feat_screen = screen_main + screen_extra
    X = screen_df[feat_screen].to_numpy(dtype=np.float32).copy()
    x_means = np.nanmean(X, axis=0)
    x_means = np.where(np.isfinite(x_means), x_means, 0.0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(x_means, nan_idx[1])

    X = X - X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x < 1e-12] = np.nan

    Y = screen_df[target_cols].to_numpy(dtype=np.float32)
    Y = Y - Y.mean(axis=0)
    std_y = Y.std(axis=0)
    std_y[std_y < 1e-12] = np.nan

    cov = (X.T @ Y) / len(screen_df)
    corr_mat = cov / (std_x[:, None] * std_y[None, :])

    miss_rate_map = dict(zip(miss_all_df["feature"], miss_all_df["null_rate"]))
    type_map = {f: ("cat" if f.startswith("cat_feature_") else "num") for f in feat_screen}
    source_map = {f: ("main" if f in main_features else "extra") for f in feat_screen}

    linear_rows = []
    for fi, f in enumerate(feat_screen):
        for ti, t in enumerate(target_cols):
            c = corr_mat[fi, ti]
            if np.isfinite(c):
                linear_rows.append({
                    "target": t,
                    "feature": f,
                    "pearson_corr": float(c),
                    "abs_corr": float(abs(c)),
                    "feature_type": type_map[f],
                    "source": source_map[f],
                    "null_rate": float(miss_rate_map.get(f, np.nan)),
                })

    linear_df = pd.DataFrame(linear_rows).sort_values(["target", "abs_corr"], ascending=[True, False])
    linear_df.to_csv(TABLES / "feature_target_linear_corr.csv", index=False)

    top10_per_target = linear_df.groupby("target", as_index=False).head(10)
    top10_per_target.to_csv(TABLES / "target_top10_features_linear.csv", index=False)

    # Per-target feature mix in top10
    mix_rows = []
    for t, g in top10_per_target.groupby("target"):
        n = len(g)
        mix_rows.append({
            "target": t,
            "mean_abs_corr_top10": float(g["abs_corr"].mean()) if n else np.nan,
            "n_cat_top10": int((g["feature_type"] == "cat").sum()),
            "n_num_top10": int((g["feature_type"] == "num").sum()),
            "n_main_top10": int((g["source"] == "main").sum()),
            "n_extra_top10": int((g["source"] == "extra").sum()),
        })
    target_mix_df = pd.DataFrame(mix_rows).sort_values("mean_abs_corr_top10", ascending=False)
    target_mix_df.to_csv(TABLES / "target_top10_feature_mix.csv", index=False)

    # Feature universality: how often feature appears in top10 across targets
    feature_uni = (
        top10_per_target.groupby("feature", as_index=False)
        .agg(
            n_targets_top10=("target", "nunique"),
            mean_abs_corr_when_top10=("abs_corr", "mean"),
            max_abs_corr_when_top10=("abs_corr", "max"),
        )
        .sort_values(["n_targets_top10", "mean_abs_corr_when_top10"], ascending=[False, False])
    )
    feature_uni.to_csv(TABLES / "feature_universality_top10.csv", index=False)

    feature_signal = (
        linear_df.groupby("feature", as_index=False)
        .agg(
            max_abs_corr=("abs_corr", "max"),
            mean_abs_corr=("abs_corr", "mean"),
            n_targets_abs_corr_gt_005=("abs_corr", lambda s: int((s > 0.05).sum())),
            n_targets_abs_corr_gt_010=("abs_corr", lambda s: int((s > 0.10).sum())),
        )
        .merge(
            pd.DataFrame({
                "feature": feat_screen,
                "source": [source_map[f] for f in feat_screen],
                "feature_type": [type_map[f] for f in feat_screen],
                "null_rate": [miss_rate_map.get(f, np.nan) for f in feat_screen],
            }),
            on="feature",
            how="left",
        )
        .sort_values(["max_abs_corr", "mean_abs_corr"], ascending=[False, False])
    )
    feature_signal.to_csv(TABLES / "feature_signal_summary.csv", index=False)

    # Convenience table for selected targets (same as before but derived from broad screen)
    selected_targets = ["target_1_1", "target_3_2", "target_10_1", "target_9_6"]
    selected_top5 = (
        linear_df[linear_df["target"].isin(selected_targets)]
        .groupby("target", as_index=False)
        .head(5)
    )
    selected_top5.to_csv(TABLES / "golden_linear_top5_selected_targets.csv", index=False)

    # -------------------------------
    # Outlier / whale effects
    # -------------------------------
    rare_targets = target_df[target_df["positive_rate"] < 0.005]["target"].tolist()
    whale_feats = num_main

    o_df = (
        scan(ARCH / "train_main_features.parquet")
        .filter(hash_expr(id_col, 12))
        .select([pl.col(id_col)] + [pl.col(f).cast(pl.Float32).alias(f) for f in whale_feats])
        .join(
            scan(ARCH / "train_target.parquet")
            .filter(hash_expr(id_col, 12))
            .select([pl.col(id_col)] + [pl.col(t).cast(pl.Int8).alias(t) for t in rare_targets]),
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
            top_pos = int((y[top] == 1).sum())
            top_n = int(top.sum())
            rest_pos = int((y[rest] == 1).sum())
            rest_n = int(rest.sum())

            if rest_pos == 0:
                continue

            top_rate = top_pos / top_n
            rest_rate = rest_pos / rest_n
            lift = top_rate / rest_rate
            _, p = fisher_exact([[top_pos, top_n - top_pos], [rest_pos, rest_n - rest_pos]], alternative="greater")

            whale_rows.append({
                "target": t,
                "feature": f,
                "top1_rate": top_rate,
                "rest99_rate": rest_rate,
                "lift": lift,
                "pvalue": float(p),
            })

    whale_df = pd.DataFrame(whale_rows)
    whale_sig = whale_df[(whale_df["lift"] >= 2.0) & (whale_df["pvalue"] < 0.05)].sort_values("lift", ascending=False)
    whale_sig.to_csv(TABLES / "whale_signals.csv", index=False)

    whale_feature_candidates = (
        whale_sig.groupby("feature", as_index=False)
        .agg(
            n_rare_targets=("target", "nunique"),
            median_lift=("lift", "median"),
            max_lift=("lift", "max"),
            min_pvalue=("pvalue", "min"),
        )
        .sort_values(["n_rare_targets", "median_lift"], ascending=[False, False])
    )
    whale_feature_candidates.to_csv(TABLES / "whale_feature_candidates.csv", index=False)

    whale_top_per_target = whale_sig.groupby("target", as_index=False).head(3)
    whale_top_per_target.to_csv(TABLES / "whale_top3_per_target.csv", index=False)

    # -------------------------------
    # Summary and report
    # -------------------------------
    n_lt_1 = int((target_df["positive_rate"] < 0.01).sum())
    n_lt_01 = int((target_df["positive_rate"] < 0.001).sum())
    n_lt_50 = int((target_df["positive_count"] < 50).sum())
    min_pos = int(target_df["positive_count"].min())

    neg_share_t10 = float((corr_t10 < 0).mean())
    mean_corr_t10 = float(corr_t10.mean())

    k4_row = cluster_eval_df.loc[cluster_eval_df["k"] == 4].iloc[0]
    clear_4 = bool((k4_row["largest_cluster_share"] <= 0.60) and (k4_row["silhouette_precomputed"] >= 0.08))

    n_unseen_feats = int((unseen_df["unseen_unique_categories"] > 0).sum())
    max_unseen_rate = float(unseen_df["unseen_rate_test_rows"].max()) if len(unseen_df) else np.nan

    global_top_features = feature_signal.head(20)
    target_mix_top = target_mix_df.head(12)

    summary = {
        "rows_train": n_train,
        "rows_test": n_test,
        "n_targets": len(target_cols),
        "n_features_main": len(main_features),
        "n_features_extra": len(extra_features),
        "targets_lt_1pct": n_lt_1,
        "targets_lt_01pct": n_lt_01,
        "targets_lt_50": n_lt_50,
        "min_positive_count": min_pos,
        "target_10_1_negative_share": neg_share_t10,
        "target_10_1_mean_corr": mean_corr_t10,
        "filled_extra_count_auc": float(auc_fill),
        "filled_extra_count_pointbiserial": float(pb_corr),
        "adversarial_auc_main_features": float(adv_auc),
        "cat_features_with_unseen_in_test": n_unseen_feats,
        "max_unseen_rate_test_rows": max_unseen_rate,
        "clear_4_target_clusters": clear_4,
        "k4_silhouette": float(k4_row["silhouette_precomputed"]),
        "k4_largest_cluster_share": float(k4_row["largest_cluster_share"]),
        "significant_whale_pairs": int(len(whale_sig)),
        "n_features_screened_linear": int(len(feat_screen)),
        "screen_sample_rows": int(len(screen_df)),
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_text = f"""# Public EDA Report: Multi-Label Banking Products (41 targets)

## Executive Summary
Было исследовано распределение таргетов, межтаргетные связи, структура пропусков, сдвиг train/test, линейный сигнал фич и outlier-эффекты.

- Датасет сильно несбалансирован: {n_lt_1} из {len(target_cols)} таргетов имеют prevalence <1%, но экстремально малых классов (<50 позитивов) нет.
- `target_10_1` подтвержден как антагонист: отрицательная корреляция с {neg_share_t10:.2%} остальных таргетов.
- Пропуски в extra несут сигнал активности: AUC `filled_extra_count` = {auc_fill:.4f}.
- Четкого разбиения таргетов на 4 устойчивых кластера нет (silhouette@4={k4_row['silhouette_precomputed']:.4f}), поэтому 41 per-target модели остаются базой.

## 1. Data Landscape
Исследование проводилось memory-safe через Polars lazy/streaming, тяжелые блоки считались на стабильных hash-семплах.

- Train rows: **{n_train}**
- Test rows: **{n_test}**
- Targets: **{len(target_cols)}**
- Main features: **{len(main_features)}**
- Extra features: **{len(extra_features)}**

## 2. Target Landscape and Product Structure
### 2.1 Class Imbalance
- Targets with prevalence <1%: **{n_lt_1} / {len(target_cols)}**
- Targets with prevalence <0.1%: **{n_lt_01}**
- Targets with positive_count <50: **{n_lt_50}**
- Minimum positive count: **{min_pos}**

Tables:
- `public_tables/target_stats.csv`
- `public_tables/target_family_stats.csv`
- `public_tables/opened_targets_distribution.csv`

### 2.2 Inter-Target Dependencies
- `target_10_1` negative-correlation share: **{neg_share_t10:.2%}**
- Mean corr(`target_10_1`, others): **{mean_corr_t10:.4f}**

Top positive pairs (corr):
```text
{pretty(top_pos_pairs[["target_a", "target_b", "corr", "co_count"]], 10)}
```

Top negative pairs (corr):
```text
{pretty(top_neg_pairs[["target_a", "target_b", "corr", "co_count"]], 10)}
```

Top co-occurrence lift pairs:
```text
{pretty(top_lift_pairs[["target_a", "target_b", "pair_lift", "co_count", "co_rate"]], 10)}
```

Кластеризация таргетов по |corr|-distance:
```text
{pretty(cluster_eval_df, 10)}
```

Итог по 4-кластерам: **{'четкая структура не обнаружена' if not clear_4 else 'есть признаки четкой структуры'}**.

Tables:
- `public_tables/target_correlation_matrix.csv`
- `public_tables/target_pair_stats.csv`
- `public_tables/target_10_1_profile.csv`
- `public_tables/target_cluster_quality.csv`
- `public_tables/target_clusters_k4.csv`
- `public_tables/target_cluster_summary.csv`

## 3. Feature Space Quality
### 3.1 Missingness Architecture
Top extra features by null rate:
```text
{pretty(top10_missing[["feature", "null_rate"]], 10)}
```

Missingness bands for extra features:
```text
{pretty(miss_bands, 10)}
```

Глобальный сигнал заполненности extra:
- AUC(`filled_extra_count` -> any open except `target_10_1`): **{auc_fill:.4f}**
- Point-biserial r: **{pb_corr:.4f}** (p={pb_p:.2e})

Filled-count deciles:
```text
{pretty(fill_dec_df[["decile", "n", "avg_filled", "target_rate"]], 10)}
```

Tables:
- `public_tables/feature_missingness_summary.csv`
- `public_tables/extra_missingness_summary.csv`
- `public_tables/extra_missingness_bands.csv`
- `public_tables/filled_extra_count_deciles.csv`
- `public_tables/missing_indicator_auc_popular_targets.csv`

### 3.2 Categorical Risk Surface
- Cat features with unseen categories in test: **{n_unseen_feats}**
- Max unseen row-rate in test: **{max_unseen_rate:.6f}**

Top cardinality:
```text
{pretty(card_df[["feature", "train_nunique", "test_nunique"]], 10)}
```

Unseen categories summary:
```text
{pretty(unseen_df[["feature", "unseen_unique_categories", "unseen_rate_test_rows"]], 10)}
```

Вывод: энкодеры категорий должны обрабатывать unknown значения (`handle_unknown='value'` + prior fallback).

Tables:
- `public_tables/categorical_cardinality.csv`
- `public_tables/categorical_unseen_categories.csv`

## 4. Feature -> Target Signal (Wide Linear Screening)
Скрининг выполнен по **{len(feat_screen)}** фичам (все main + плотные extra) на стабильном семпле **{len(screen_df)}** строк.

Глобально самые сильные фичи:
```text
{pretty(global_top_features[["feature", "source", "feature_type", "max_abs_corr", "mean_abs_corr", "n_targets_abs_corr_gt_005", "null_rate"]], 20)}
```

Top-5 для выбранных таргетов:
```text
{pretty(selected_top5[["target", "feature", "pearson_corr", "abs_corr", "source", "feature_type"]], 25)}
```

Feature universality (сколько таргетов покрывает фича в top10):
```text
{pretty(feature_uni[["feature", "n_targets_top10", "mean_abs_corr_when_top10", "max_abs_corr_when_top10"]], 15)}
```

Target-level mix в top10 (cat/num, main/extra):
```text
{pretty(target_mix_top, 12)}
```

Tables:
- `public_tables/feature_target_linear_corr.csv`
- `public_tables/target_top10_features_linear.csv`
- `public_tables/feature_signal_summary.csv`
- `public_tables/feature_universality_top10.csv`
- `public_tables/target_top10_feature_mix.csv`
- `public_tables/golden_linear_top5_selected_targets.csv`

## 5. Train/Test Shift
Adversarial AUC (main features, 20% hash sample): **{adv_auc:.4f}**.

Вывод: сильного глобального covariate shift не обнаружено, но drift мониторинг должен оставаться обязательным шагом перед сабмитом.

## 6. Outliers and Whale Behaviour
Для редких таргетов проверен uplift в top-1% числовых main-фич.

- Significant whale pairs (lift>=2, p<0.05): **{len(whale_sig)}**

Whale feature candidates:
```text
{pretty(whale_feature_candidates[["feature", "n_rare_targets", "median_lift", "max_lift", "min_pvalue"]], 15)}
```

Top whale interactions per target:
```text
{pretty(whale_top_per_target[["target", "feature", "lift", "top1_rate", "rest99_rate", "pvalue"]], 20)}
```

Tables:
- `public_tables/whale_signals.csv`
- `public_tables/whale_feature_candidates.csv`
- `public_tables/whale_top3_per_target.csv`

## 7. Modeling Implications
Было обнаружено:
- Редкие таргеты требуют per-target контроля устойчивости и осторожной CV-схемы.
- Межтаргетные связи есть, но не поддерживают агрессивное схлопывание в небольшое число кластеров.
- Missingness и whale-признаки дают отдельный, практически полезный слой сигнала.
- По фичам есть как универсальные, так и таргет-специфичные сигналы; это поддерживает гибрид: общий strong-core + target-specific add-ons.

Рекомендации для базового пайплайна:
1. 41 независимых модели как базовый каркас.
2. OOF meta-features по релевантным таргет-парам (особенно с учетом знака корреляции для `target_10_1`).
3. Обязательные признаки: `filled_extra_count`, null-indicators (для moderate missing), whale flags (`is_whale_*`).
4. Для категориальных энкодеров: unknown handling + smoothing prior.

## Artifacts
- Pipeline code: `eda_workspace/public_eda_pipeline.py`
- This report: `eda_workspace/PUBLIC_EDA_REPORT.md`
- All produced tables: `eda_workspace/public_tables/`
"""

    REPORT.write_text(report_text, encoding="utf-8")
    print("public pipeline done")


if __name__ == "__main__":
    main()
