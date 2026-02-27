# Public EDA Report: Multi-Label Banking Products (41 targets)

## Executive Summary
Было исследовано распределение таргетов, межтаргетные связи, структура пропусков, сдвиг train/test, линейный сигнал фич и outlier-эффекты.

- Датасет сильно несбалансирован: 24 из 41 таргетов имеют prevalence <1%, но экстремально малых классов (<50 позитивов) нет.
- `target_10_1` подтвержден как антагонист: отрицательная корреляция с 100.00% остальных таргетов.
- Пропуски в extra несут сигнал активности: AUC `filled_extra_count` = 0.6520.
- Четкого разбиения таргетов на 4 устойчивых кластера нет (silhouette@4=0.0148), поэтому 41 per-target модели остаются базой.

## 1. Data Landscape
Исследование проводилось memory-safe через Polars lazy/streaming, тяжелые блоки считались на стабильных hash-семплах.

- Train rows: **750000**
- Test rows: **250000**
- Targets: **41**
- Main features: **199**
- Extra features: **2241**

## 2. Target Landscape and Product Structure
### 2.1 Class Imbalance
- Targets with prevalence <1%: **24 / 41**
- Targets with prevalence <0.1%: **3**
- Targets with positive_count <50: **0**
- Minimum positive count: **83**

Tables:
- `public_tables/target_stats.csv`
- `public_tables/target_family_stats.csv`
- `public_tables/opened_targets_distribution.csv`

### 2.2 Inter-Target Dependencies
- `target_10_1` negative-correlation share: **100.00%**
- Mean corr(`target_10_1`, others): **-0.0849**

Top positive pairs (corr):
```text
  target_a   target_b     corr  co_count
target_5_1 target_5_2 0.517938      1906
target_6_1 target_6_4 0.514028      3236
target_6_4 target_6_5 0.265718       419
target_1_4 target_2_2 0.218648      4344
target_6_1 target_6_2 0.190234      1192
target_6_4 target_8_1 0.188547      4390
target_2_5 target_2_6 0.155876       343
target_3_2 target_5_1 0.130180      3468
target_6_3 target_7_1 0.128541      2045
target_2_4 target_9_3 0.119986      1162
```

Top negative pairs (corr):
```text
  target_a    target_b      corr  co_count
target_9_6 target_10_1 -0.363408         0
target_8_1 target_10_1 -0.229191         0
target_3_1 target_10_1 -0.224020         0
target_3_2 target_10_1 -0.222801         0
target_9_7 target_10_1 -0.196229         0
target_7_1 target_10_1 -0.175112         0
target_9_2 target_10_1 -0.131908         0
target_8_2 target_10_1 -0.124360         0
target_7_2 target_10_1 -0.114413         0
target_2_2 target_10_1 -0.109367         0
```

Top co-occurrence lift pairs:
```text
  target_a   target_b  pair_lift  co_count  co_rate
target_6_4 target_6_5 127.312850       419 0.000559
target_5_1 target_5_2 106.295552      1906 0.002541
target_6_1 target_6_4  62.205101      3236 0.004315
target_2_5 target_2_6  54.775940       343 0.000457
target_1_3 target_2_7  28.522371       154 0.000205
target_6_1 target_6_2  24.360972      1192 0.001589
target_5_2 target_7_3  13.624991       111 0.000148
target_2_4 target_2_5  12.551108       135 0.000180
target_2_4 target_2_6  12.511669       313 0.000417
target_6_3 target_6_4  11.930097       408 0.000544
```

Кластеризация таргетов по |corr|-distance:
```text
 k  silhouette_precomputed  largest_cluster_share  min_cluster_size  max_cluster_size
 3                0.015579               0.902439                 1                37
 4                0.014810               0.878049                 1                36
 5                0.015017               0.829268                 1                34
```

Итог по 4-кластерам: **четкая структура не обнаружена**.

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
         feature  null_rate
 num_feature_923   0.999997
num_feature_1792   0.999995
num_feature_1058   0.999991
num_feature_1832   0.999991
num_feature_1695   0.999989
 num_feature_265   0.999988
num_feature_1922   0.999983
num_feature_1054   0.999980
num_feature_1277   0.999971
num_feature_1617   0.999967
```

Missingness bands for extra features:
```text
  band  count
  >99%    296
  >95%    499
  >90%    571
50-90%    553
10-50%   1096
 <=10%     21
```

Глобальный сигнал заполненности extra:
- AUC(`filled_extra_count` -> any open except `target_10_1`): **0.6520**
- Point-biserial r: **0.2509** (p=0.00e+00)

Filled-count deciles:
```text
 decile     n  avg_filled  target_rate
      0 75278  188.091886     0.456641
      1 75010  493.785149     0.551033
      2 75189  690.789464     0.597933
      3 75355  841.574759     0.646871
      4 74547  945.916791     0.708560
      5 75181 1057.726061     0.728442
      6 75122 1156.887516     0.739743
      7 74804 1237.817550     0.766403
      8 74629 1317.268703     0.812218
      9 74885 1458.716659     0.843934
```

Tables:
- `public_tables/feature_missingness_summary.csv`
- `public_tables/extra_missingness_summary.csv`
- `public_tables/extra_missingness_bands.csv`
- `public_tables/filled_extra_count_deciles.csv`
- `public_tables/missing_indicator_auc_popular_targets.csv`

### 3.2 Categorical Risk Surface
- Cat features with unseen categories in test: **2**
- Max unseen row-rate in test: **0.000184**

Top cardinality:
```text
       feature  train_nunique  test_nunique
cat_feature_39           1989          1895
cat_feature_34            120            80
cat_feature_52             51            51
cat_feature_56              5             5
 cat_feature_9              5             5
 cat_feature_6              4             4
cat_feature_40              4             4
cat_feature_45              3             3
cat_feature_49              3             3
cat_feature_48              3             3
```

Unseen categories summary:
```text
       feature  unseen_unique_categories  unseen_rate_test_rows
cat_feature_39                        38               0.000184
cat_feature_34                        13               0.000052
cat_feature_51                         0               0.000000
cat_feature_37                         0               0.000000
cat_feature_38                         0               0.000000
cat_feature_40                         0               0.000000
cat_feature_41                         0               0.000000
cat_feature_42                         0               0.000000
cat_feature_43                         0               0.000000
cat_feature_44                         0               0.000000
```

Вывод: энкодеры категорий должны обрабатывать unknown значения (`handle_unknown='value'` + prior fallback).

Tables:
- `public_tables/categorical_cardinality.csv`
- `public_tables/categorical_unseen_categories.csv`

## 4. Feature -> Target Signal (Wide Linear Screening)
Скрининг выполнен по **519** фичам (все main + плотные extra) на стабильном семпле **89728** строк.

Глобально самые сильные фичи:
```text
         feature source feature_type  max_abs_corr  mean_abs_corr  n_targets_abs_corr_gt_005  null_rate
num_feature_1263  extra          num      0.320016       0.020857                          3   0.009647
   num_feature_7   main          num      0.313191       0.024684                          5   0.297049
num_feature_1984  extra          num      0.308372       0.030979                          6   0.009647
  num_feature_33   main          num      0.293924       0.021599                          3   0.392892
  num_feature_41   main          num      0.236630       0.030782                          7   0.064324
 num_feature_787  extra          num      0.229101       0.021208                          3   0.004287
  num_feature_36   main          num      0.228627       0.035013                          8   0.162437
num_feature_1912  extra          num      0.225507       0.021981                          3   0.004287
  cat_feature_39   main          cat      0.223694       0.025688                          5   0.000000
 num_feature_176  extra          num      0.205350       0.015973                          3   0.240828
  cat_feature_48   main          cat      0.199690       0.028287                          4   0.000000
  cat_feature_56   main          cat      0.195051       0.027023                          4   0.000000
   cat_feature_9   main          cat      0.191559       0.025526                          3   0.000000
num_feature_2220  extra          num      0.190283       0.024411                          5   0.162437
num_feature_2370  extra          num      0.183689       0.022217                          6   0.004287
 num_feature_201  extra          num      0.181877       0.016809                          3   0.240828
 num_feature_342  extra          num      0.178265       0.019292                          3   0.214536
 num_feature_367  extra          num      0.177310       0.029323                          5   0.240828
  cat_feature_46   main          cat      0.174428       0.021639                          4   0.000000
num_feature_2020  extra          num      0.172704       0.013980                          2   0.240828
```

Top-5 для выбранных таргетов:
```text
     target          feature  pearson_corr  abs_corr source feature_type
target_10_1   cat_feature_48      0.199690  0.199690   main          cat
target_10_1   cat_feature_46      0.174428  0.174428   main          cat
target_10_1   cat_feature_52      0.172027  0.172027   main          cat
target_10_1   cat_feature_21      0.169186  0.169186   main          cat
target_10_1   num_feature_36     -0.162721  0.162721   main          num
 target_1_1   num_feature_67      0.088394  0.088394   main          num
 target_1_1 num_feature_1928      0.062740  0.062740  extra          num
 target_1_1 num_feature_2114      0.052991  0.052991  extra          num
 target_1_1  num_feature_109      0.052075  0.052075   main          num
 target_1_1   num_feature_81      0.051993  0.051993   main          num
 target_3_2 num_feature_1984      0.308372  0.308372  extra          num
 target_3_2   num_feature_41      0.236630  0.236630   main          num
 target_3_2   cat_feature_31      0.128868  0.128868   main          cat
 target_3_2   cat_feature_14      0.128841  0.128841   main          cat
 target_3_2   cat_feature_29      0.128841  0.128841   main          cat
 target_9_6   cat_feature_48     -0.122887  0.122887   main          cat
 target_9_6   cat_feature_49     -0.111562  0.111562   main          cat
 target_9_6   cat_feature_44     -0.109900  0.109900   main          cat
 target_9_6   cat_feature_13     -0.108934  0.108934   main          cat
 target_9_6   cat_feature_26     -0.108934  0.108934   main          cat
```

Feature universality (сколько таргетов покрывает фича в top10):
```text
         feature  n_targets_top10  mean_abs_corr_when_top10  max_abs_corr_when_top10
  num_feature_36               11                  0.084307                 0.228627
  num_feature_41               11                  0.058771                 0.236630
  cat_feature_56               11                  0.054742                 0.195051
num_feature_1984                8                  0.096909                 0.308372
num_feature_2350                8                  0.066028                 0.137630
num_feature_2064                8                  0.061266                 0.128301
  cat_feature_48                7                  0.084754                 0.199690
 num_feature_524                7                  0.064484                 0.123385
  cat_feature_17                6                  0.095155                 0.153674
  cat_feature_39                6                  0.080149                 0.223694
   cat_feature_9                6                  0.064803                 0.191559
 num_feature_116                6                  0.060944                 0.139486
  cat_feature_40                6                  0.036876                 0.069329
   num_feature_7                5                  0.100956                 0.313191
 num_feature_130                5                  0.079332                 0.126769
```

Target-level mix в top10 (cat/num, main/extra):
```text
     target  mean_abs_corr_top10  n_cat_top10  n_num_top10  n_main_top10  n_extra_top10
 target_8_1             0.230803            3            7             6              4
target_10_1             0.166684            9            1            10              0
 target_2_2             0.158813            8            2             9              1
 target_3_2             0.157572            8            2             9              1
 target_1_3             0.131995            8            2            10              0
 target_9_2             0.116695            1            9             3              7
 target_9_6             0.110398           10            0            10              0
 target_9_8             0.109921            0           10             4              6
 target_9_7             0.109456            2            8             5              5
 target_1_4             0.095712           10            0            10              0
 target_3_4             0.094336            0           10             7              3
 target_7_1             0.090488            1            9             5              5
```

Tables:
- `public_tables/feature_target_linear_corr.csv`
- `public_tables/target_top10_features_linear.csv`
- `public_tables/feature_signal_summary.csv`
- `public_tables/feature_universality_top10.csv`
- `public_tables/target_top10_feature_mix.csv`
- `public_tables/golden_linear_top5_selected_targets.csv`

## 5. Train/Test Shift
Adversarial AUC (main features, 20% hash sample): **0.5007**.

Вывод: сильного глобального covariate shift не обнаружено, но drift мониторинг должен оставаться обязательным шагом перед сабмитом.

## 6. Outliers and Whale Behaviour
Для редких таргетов проверен uplift в top-1% числовых main-фич.

- Significant whale pairs (lift>=2, p<0.05): **313**

Whale feature candidates:
```text
        feature  n_rare_targets  median_lift  max_lift   min_pvalue
num_feature_131               7     4.785611  8.465113 7.774070e-10
 num_feature_33               6     8.046331 18.508651 1.759406e-08
 num_feature_16               6     6.676415 67.362039 2.417461e-40
  num_feature_1               6     2.230901  2.668370 1.263853e-07
num_feature_102               6     2.230901  2.668370 1.263853e-07
num_feature_106               6     2.230901  2.668370 1.263853e-07
num_feature_124               6     2.230901  2.668370 1.263853e-07
 num_feature_14               6     2.230901  2.668370 1.263853e-07
 num_feature_20               6     2.230901  2.668370 1.263853e-07
 num_feature_32               6     2.230901  2.668370 1.263853e-07
 num_feature_79               6     2.230901  2.668370 1.263853e-07
 num_feature_80               6     2.230901  2.668370 1.263853e-07
 num_feature_86               6     2.230901  2.668370 1.263853e-07
  num_feature_9               6     2.230901  2.668370 1.263853e-07
num_feature_117               5     8.811568 24.636013 2.111487e-23
```

Top whale interactions per target:
```text
    target         feature      lift  top1_rate  rest99_rate       pvalue
target_3_4  num_feature_16 67.362039   0.003385     0.000050 2.417461e-40
target_6_5  num_feature_57 45.487062   0.027397     0.000602 9.771675e-04
target_3_5  num_feature_23 43.457282   0.058252     0.001340 4.250825e-16
target_3_5  num_feature_71 43.120668   0.059211     0.001373 1.919786e-12
target_3_5  num_feature_76 41.050626   0.043429     0.001058 1.429171e-44
target_6_5  num_feature_25 32.327555   0.018779     0.000581 1.029188e-05
target_6_5  num_feature_87 29.597945   0.016241     0.000549 1.064032e-08
target_2_7  num_feature_42 27.431373   0.005208     0.000190 3.782731e-02
target_2_7   num_feature_8 26.867270   0.004525     0.000168 3.016803e-04
target_3_4  num_feature_88 24.142934   0.043689     0.001810 2.997959e-10
target_3_4  num_feature_23 24.142934   0.043689     0.001810 2.997959e-10
target_2_3  num_feature_56 14.164523   0.019337     0.001365 1.075883e-06
target_9_4 num_feature_103 13.410538   0.021711     0.001619 9.953251e-14
target_2_7  num_feature_58 12.678653   0.002283     0.000180 1.313284e-02
target_3_3 num_feature_127 12.038710   0.012500     0.001038 1.265756e-02
target_9_4 num_feature_127 10.629114   0.018750     0.001764 3.090680e-03
target_7_3  num_feature_54  9.589041   0.039062     0.004074 2.026614e-04
target_1_5 num_feature_101  9.396618   0.002400     0.000255 9.727506e-15
target_1_5 num_feature_113  9.396618   0.002400     0.000255 9.727506e-15
target_3_3  num_feature_23  9.345652   0.009709     0.001039 2.036305e-02
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
