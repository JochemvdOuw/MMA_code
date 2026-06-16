import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# DATA LOADING
# =========================================================

index_names = ['engine', 'cycle']
operational_condition_names = ['altitude', 'mach_nr', 'TRA']
sensor_names = [
    'T2',        # total temperature at fan inlet
    'T24',       # total temperature at LPC outlet
    'T30',       # total temperature at HPC outlet
    'T50',       # total temperature at LPT outlet
    'P2',        # pressure at fan inlet
    'P15',       # total pressure in bypass-duct
    'P30',       # total pressure at HPC outlet
    'Nf',        # physical fan speed (rpm)
    'Nc',        # physical core speed (rpm)
    'epr',       # engine pressure ratio (P50/P2)
    'Ps30',      # static pressure at HPC outlet
    'phi',       # ratio of fuel flow to Ps30
    'NRf',       # corrected fan speed
    'NRc',       # corrected core speed
    'BPR',       # bypass ratio
    'farB',      # burner fuel-air ratio
    'htBleed',   # bleed enthalpy
    'Nf_dmd',    # demanded fan speed (rpm)
    'PCNfR_dmd', # demanded corrected fan speed (rpm)
    'W31',       # HPT coolant bleed (lbm/s)
    'W32',       # LPT coolant bleed
]
col_names = index_names + operational_condition_names + sensor_names

df_train = pd.read_csv(r'train_FD001.txt', sep=' ', names=col_names,
                       index_col=False, usecols=range(len(col_names)))
df_test  = pd.read_csv(r'test_FD001.txt',  sep=' ', names=col_names,
                       index_col=False, usecols=range(len(col_names)))
df_rul   = pd.read_csv(r'RUL_FD001.txt', header=None, names=['RUL'])


# =========================================================
# SECTION 2.1 — DATA EXPLORATION
# =========================================================

RUL_CAP = 125  # engines with RUL > 125 are considered "healthy"; cap at 125

# --- Add capped RUL column to training data ---
max_cycles = df_train.groupby('engine')['cycle'].max().reset_index()
max_cycles.columns = ['engine', 'max_cycle']
df_train = df_train.merge(max_cycles, on='engine')
df_train['RUL'] = (df_train['max_cycle'] - df_train['cycle']).clip(upper=RUL_CAP)
df_train.drop('max_cycle', axis=1, inplace=True)

print("=== Dataset overview ===")
print(f"Training samples : {len(df_train):,}  rows  |  {df_train['engine'].nunique()} engines")
print(f"Test samples     : {len(df_test):,}  rows  |  {df_test['engine'].nunique()} engines")
print(f"RUL range (train): {df_train['RUL'].min()} – {df_train['RUL'].max()} cycles\n")

# -------------------------------------------------------
# 2.1.1  Identify near-constant (uninformative) features
# -------------------------------------------------------
all_features = operational_condition_names + sensor_names
feature_std = df_train[all_features].std().sort_values()

print("=== Standard deviation of all features (sorted) ===")
print(feature_std.to_string())

STD_THRESHOLD = 0.5  # features below this are effectively constant for FD001
drop_cols  = feature_std[feature_std < STD_THRESHOLD].index.tolist()
keep_sensors = [s for s in sensor_names if s not in drop_cols]

print(f"\n>>> Features dropped (std < {STD_THRESHOLD}): {drop_cols}")
print(f">>> Sensors kept ({len(keep_sensors)}): {keep_sensors}\n")

# Bar chart of standard deviations to make the gap visible
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#d62728' if f in drop_cols else '#1f77b4' for f in feature_std.index]
ax.bar(feature_std.index, feature_std.values, color=colors)
ax.axhline(STD_THRESHOLD, color='red', linestyle='--', label=f'Drop threshold (std={STD_THRESHOLD})')
ax.set_xlabel('Feature')
ax.set_ylabel('Standard deviation')
ax.set_title('Feature variability — red bars are dropped (near-constant)')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Feature_Variability.png', dpi=150)
plt.show(block=False)

# -------------------------------------------------------
# 2.1.2  Sensor trend plots — degradation over time
# -------------------------------------------------------
sample_engines = [1, 2, 3, 4, 5]
n_sensors = len(keep_sensors)
ncols = 3
nrows = int(np.ceil(n_sensors / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), sharex=False)
axes = axes.flatten()

for i, sensor in enumerate(keep_sensors):
    ax = axes[i]
    for eng in sample_engines:
        eng_data = df_train[df_train['engine'] == eng].sort_values('cycle')
        ax.plot(eng_data['cycle'], eng_data[sensor], alpha=0.7, linewidth=0.9)
    ax.set_title(sensor, fontsize=9)
    ax.set_xlabel('Cycle', fontsize=8)
    ax.tick_params(labelsize=7)

# hide unused subplots
for j in range(n_sensors, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Sensor trends over time (engines 1–5)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('Sensor_Trends.png', dpi=150, bbox_inches='tight')
plt.show(block=False)

# -------------------------------------------------------
# 2.1.3  Correlation heatmap — sensors vs. RUL
# -------------------------------------------------------
corr_cols = keep_sensors + ['RUL']
corr = df_train[corr_cols].corr()

fig, ax = plt.subplots(figsize=(len(corr_cols) * 0.7 + 2, len(corr_cols) * 0.7 + 1))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.4, annot_kws={'size': 7}, ax=ax)
ax.set_title('Pearson correlation — kept sensors + RUL')
plt.tight_layout()
plt.savefig('Correlation_Heatmap.png', dpi=150, bbox_inches='tight')
plt.show(block=True)

print("=== Sensors most correlated with RUL ===")
rul_corr = corr['RUL'].drop('RUL').sort_values(key=abs, ascending=False)
print(rul_corr.to_string())

# Top-8 by |Pearson correlation| with RUL — confirmed from heatmap above
top8_sensors = ['T50', 'phi', 'P30', 'htBleed', 'T24', 'T30', 'Nc', 'NRc']
print(f"\nSensors for modelling: {top8_sensors}")


# =========================================================
# SECTION 2.2 — FULL ML PIPELINE
# =========================================================
# pip install xgboost   (if not yet installed)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBRegressor

# STEP 1 — rolling mean, std, and EWM per sensor over the last W cycles
def make_rolling_features(df, sensors, window, min_periods=None):
    if min_periods is None:
        min_periods = window
    parts = [df[['engine', 'cycle']].reset_index(drop=True)]
    for s in sensors:
        g = df.groupby('engine')[s]
        parts.append(
            g.transform(lambda x: x.rolling(window, min_periods=min_periods).mean())
             .rename(f'{s}_mean').reset_index(drop=True)
        )
        parts.append(
            g.transform(lambda x: x.rolling(window, min_periods=min_periods).std().fillna(0))
             .rename(f'{s}_std').reset_index(drop=True)
        )
        # EWM: weights recent cycles more heavily (α = 2/(span+1))
        parts.append(
            g.transform(lambda x: x.ewm(span=window, min_periods=1).mean())
             .rename(f'{s}_ewm').reset_index(drop=True)
        )
    result = pd.concat(parts, axis=1)
    if 'RUL' in df.columns:
        result['RUL'] = df['RUL'].reset_index(drop=True)
    return result

# Build features with initial window W=30 (refined in Section 2.3)
WINDOW = 30
feat_train = make_rolling_features(df_train, top8_sensors, WINDOW).dropna()
feat_test_raw  = make_rolling_features(df_test, top8_sensors, WINDOW, min_periods=1)
feat_test_last = feat_test_raw.groupby('engine').last().reset_index()

feature_cols = [c for c in feat_train.columns if c not in ('engine', 'cycle', 'RUL')]
print(f"\nFeature vector: {len(feature_cols)} features "
      f"({len(top8_sensors)} sensors × 3 statistics: mean + std + ewm)")

# STEP 2 — split by engine ID so no engine appears in both train and val
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(feat_train, groups=feat_train['engine']))

X_tr = feat_train.iloc[tr_idx][feature_cols].values
y_tr = feat_train.iloc[tr_idx]['RUL'].values
X_va = feat_train.iloc[va_idx][feature_cols].values
y_va = feat_train.iloc[va_idx]['RUL'].values

n_tr_eng = feat_train.iloc[tr_idx]['engine'].nunique()
n_va_eng = feat_train.iloc[va_idx]['engine'].nunique()
print(f"Train : {len(X_tr):,} rows ({n_tr_eng} engines)  |  "
      f"Val: {len(X_va):,} rows ({n_va_eng} engines)")

# STEP 3 — z-score normalisation, scaler fit on training rows only
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(feat_test_last[feature_cols].values)

# STEP 4 — drop features below median ExtraTrees importance
selector = SelectFromModel(
    ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    threshold='median'
)
X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
X_va_sel = selector.transform(X_va_s)
X_te_sel = selector.transform(X_te_s)

selected_features = [f for f, keep in zip(feature_cols, selector.get_support()) if keep]
print(f"Features selected ({len(selected_features)}): {selected_features}\n")

# STEP 5A — Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                            min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_tr_sel, y_tr)
y_va_rf = np.clip(rf.predict(X_va_sel), 0, RUL_CAP)
rmse_rf = float(np.sqrt(np.mean((y_va - y_va_rf) ** 2)))
print(f"Random Forest  — Validation RMSE: {rmse_rf:.2f} cycles")

# STEP 5B — XGBoost (hyperparameters refined in §2.3)
xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, reg_lambda=5, reg_alpha=0.1,
                          random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_tr_sel, y_tr)
y_va_xgb = np.clip(xgb_model.predict(X_va_sel), 0, RUL_CAP)
rmse_xgb = float(np.sqrt(np.mean((y_va - y_va_xgb) ** 2)))
print(f"XGBoost        — Validation RMSE: {rmse_xgb:.2f} cycles")

# ----------------------------------------------------------
# STEP 6 — Validation scatter plots (predicted vs true RUL)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, label, rmse in zip(
        axes,
        [y_va_rf, y_va_xgb],
        ['Random Forest', 'XGBoost'],
        [rmse_rf, rmse_xgb]):
    ax.scatter(y_va, preds, alpha=0.25, s=8)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], 'r--', lw=1.2, label='Perfect prediction')
    ax.set_xlabel('True RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.set_title(f'{label}  (RMSE = {rmse:.1f} cycles)')
    ax.set_xlim(0, RUL_CAP); ax.set_ylim(0, RUL_CAP)
    ax.legend(fontsize=8)
plt.suptitle('Validation set — predicted vs true RUL', fontsize=11)
plt.tight_layout()
plt.savefig('RUL_Validation.png', dpi=150, bbox_inches='tight')
plt.show(block=False)

# SelectFromModel keeps feature names intact, so importances map directly
importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot.barh(ax=ax, color='steelblue')
ax.set_xlabel('Mean decrease in impurity (normalised)')
ax.set_title('Random Forest — Feature importance')
plt.tight_layout()
plt.savefig('Feature_Importance.png', dpi=150)
plt.show(block=False)


# =========================================================
# SECTION 2.3 — HYPERPARAMETER TUNING  (GroupKFold CV)
# =========================================================
# Stage A: grid search over window size w (GroupKFold, 5 folds).
# Stage B: RandomizedSearchCV over model params, Pipeline ensures no fold leakage.
# Final:   both models retrained on all 100 engines, evaluated on test set.

gkf = GroupKFold(n_splits=5)
WINDOW_CANDIDATES = [50, 75, 100]

def cv_rmse_window(window, base_model):
    """GroupKFold RMSE for a given window + model (scaler+selector fitted per fold)."""
    feat = make_rolling_features(df_train, top8_sensors, window).dropna()
    fc   = [c for c in feat.columns if c not in ('engine', 'cycle', 'RUL')]
    X    = feat[fc].values
    y    = feat['RUL'].values
    grp  = feat['engine'].values
    rmses = []
    for tr_i, va_i in gkf.split(X, y, grp):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X[tr_i])
        X_va_s = sc.transform(X[va_i])
        sel = SelectFromModel(
            ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            threshold='median'
        )
        X_tr_sel = sel.fit_transform(X_tr_s, y[tr_i])
        X_va_sel = sel.transform(X_va_s)
        m = clone(base_model)
        m.fit(X_tr_sel, y[tr_i])
        preds = np.clip(m.predict(X_va_sel), 0, RUL_CAP)
        rmses.append(float(np.sqrt(np.mean((y[va_i] - preds) ** 2))))
    return float(np.mean(rmses))

# -------------------------------------------------------
# Stage A: Window size search
# -------------------------------------------------------
print("\n=== Stage A: Window size search ===")

rf_probe  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_probe = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)

print("  RF  (mean GroupKFold RMSE per window):")
rf_w = {w: cv_rmse_window(w, rf_probe) for w in WINDOW_CANDIDATES}
for w, r in rf_w.items():
    print(f"    w = {w:2d}  →  {r:.2f} cycles")
best_w_rf = min(rf_w, key=rf_w.get)
print(f"  Best window RF : {best_w_rf} cycles\n")

print("  XGB (mean GroupKFold RMSE per window):")
xgb_w = {w: cv_rmse_window(w, xgb_probe) for w in WINDOW_CANDIDATES}
for w, r in xgb_w.items():
    print(f"    w = {w:2d}  →  {r:.2f} cycles")
best_w_xgb = min(xgb_w, key=xgb_w.get)
print(f"  Best window XGB: {best_w_xgb} cycles\n")

# -------------------------------------------------------
# Stage B: Model hyperparameter search
# -------------------------------------------------------
print("=== Stage B: Model hyperparameter search ===")

def get_raw_features(window):
    """Return raw (unscaled) rolling features for all 100 training engines."""
    feat = make_rolling_features(df_train, top8_sensors, window).dropna()
    fc   = [c for c in feat.columns if c not in ('engine', 'cycle', 'RUL')]
    return feat[fc].values, feat['RUL'].values, feat['engine'].values, fc

X_rf_raw,  y_rf_all,  grp_rf,  fc_rf  = get_raw_features(best_w_rf)
X_xgb_raw, y_xgb_all, grp_xgb, fc_xgb = get_raw_features(best_w_xgb)

# --- RF pipeline search ---
pipe_rf = Pipeline([
    ('scaler',   StandardScaler()),
    ('selector', SelectFromModel(
                     ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                     threshold='median')),
    ('model',    RandomForestRegressor(random_state=42, n_jobs=-1)),
])
param_rf = {
    'model__n_estimators'    : [100, 200, 300, 500],
    'model__max_depth'       : [5, 10, 15, 20, None],
    'model__min_samples_leaf': [1, 3, 5, 10],
    'model__max_features'    : ['sqrt', 'log2', 0.5],
}
search_rf = RandomizedSearchCV(
    pipe_rf, param_distributions=param_rf,
    n_iter=30, cv=gkf,
    scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=1,
)
search_rf.fit(X_rf_raw, y_rf_all, groups=grp_rf)
print(f"\nRF  best CV RMSE : {-search_rf.best_score_:.2f} cycles")
print(f"RF  best params  : {search_rf.best_params_}")

# --- XGB pipeline search ---
pipe_xgb = Pipeline([
    ('scaler',   StandardScaler()),
    ('selector', SelectFromModel(
                     ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                     threshold='median')),
    ('model',    XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)),
])
param_xgb = {
    'model__n_estimators' : [100, 200, 300, 500],
    'model__max_depth'    : [3, 4, 5, 6],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample'    : [0.6, 0.8, 1.0],
    'model__reg_lambda'   : [1, 5, 10],
    'model__reg_alpha'    : [0, 0.1, 1.0],
}
search_xgb = RandomizedSearchCV(
    pipe_xgb, param_distributions=param_xgb,
    n_iter=30, cv=gkf,
    scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=1,
)
search_xgb.fit(X_xgb_raw, y_xgb_all, groups=grp_xgb)
print(f"\nXGB best CV RMSE : {-search_xgb.best_score_:.2f} cycles")
print(f"XGB best params  : {search_xgb.best_params_}")

# -------------------------------------------------------
# Final: refit best pipeline on all 100 training engines
# -------------------------------------------------------
print("\n=== Final models: retrain on all 100 engines ===")

rf_final  = clone(search_rf.best_estimator_)
rf_final.fit(X_rf_raw, y_rf_all)

xgb_final = clone(search_xgb.best_estimator_)
xgb_final.fit(X_xgb_raw, y_xgb_all)

# Prepare test features using each model's best window
feat_te_rf  = (make_rolling_features(df_test, top8_sensors, best_w_rf,  min_periods=1)
               .groupby('engine').last().reset_index())
feat_te_xgb = (make_rolling_features(df_test, top8_sensors, best_w_xgb, min_periods=1)
               .groupby('engine').last().reset_index())

rul_true     = df_rul['RUL'].values
rul_pred_rf  = np.clip(rf_final.predict(feat_te_rf[fc_rf].values),   0, RUL_CAP)
rul_pred_xgb = np.clip(xgb_final.predict(feat_te_xgb[fc_xgb].values), 0, RUL_CAP)

rmse_test_rf  = float(np.sqrt(np.mean((rul_true - rul_pred_rf)  ** 2)))
rmse_test_xgb = float(np.sqrt(np.mean((rul_true - rul_pred_xgb) ** 2)))
print(f"Test RMSE — Random Forest : {rmse_test_rf:.2f} cycles")
print(f"Test RMSE — XGBoost       : {rmse_test_xgb:.2f} cycles")

# Final side-by-side scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, label, rmse in zip(
        axes,
        [rul_pred_rf, rul_pred_xgb],
        [f'Random Forest  (w={best_w_rf})', f'XGBoost  (w={best_w_xgb})'],
        [rmse_test_rf, rmse_test_xgb]):
    ax.scatter(rul_true, preds, alpha=0.75, edgecolors='k', linewidths=0.3, s=50)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], 'r--', lw=1.2, label='Perfect prediction')
    ax.set_xlabel('True RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.set_title(f'{label}\nTest RMSE = {rmse:.1f} cycles')
    ax.set_xlim(0, RUL_CAP); ax.set_ylim(0, RUL_CAP)
    ax.legend(fontsize=8)
plt.suptitle('Test set — final predictions (100 engines)', fontsize=11)
plt.tight_layout()
plt.savefig('RUL_Test_Predictions.png', dpi=150, bbox_inches='tight')
plt.show(block=True)


# =========================================================
# SECTION 2.4 — RESULTS SUMMARY
# =========================================================

# -------------------------------------------------------
# 2.4.1  Comparison table (console)
# -------------------------------------------------------
# simple average ensemble; blending the two models reduces variance
rul_pred_ens = np.clip(0.5 * rul_pred_rf + 0.5 * rul_pred_xgb, 0, RUL_CAP)
rmse_test_ens = float(np.sqrt(np.mean((rul_true - rul_pred_ens) ** 2)))

mae_rf  = float(np.mean(np.abs(rul_true - rul_pred_rf)))
mae_xgb = float(np.mean(np.abs(rul_true - rul_pred_xgb)))
mae_ens = float(np.mean(np.abs(rul_true - rul_pred_ens)))

print("\n" + "=" * 72)
print(f"{'MODEL COMPARISON — TEST SET (100 engines)':^72}")
print("=" * 72)
print(f"{'Metric':<28} {'Random Forest':>14} {'XGBoost':>14} {'Ensemble':>14}")
print("-" * 72)
print(f"{'Best window (cycles)':<28} {best_w_rf:>14} {best_w_xgb:>14} {'(avg)':>14}")
print(f"{'Best CV RMSE (cycles)':<28} {-search_rf.best_score_:>14.2f} {-search_xgb.best_score_:>14.2f} {'—':>14}")
print(f"{'Test RMSE  (cycles)':<28} {rmse_test_rf:>14.2f} {rmse_test_xgb:>14.2f} {rmse_test_ens:>14.2f}")
print(f"{'Test MAE   (cycles)':<28} {mae_rf:>14.2f} {mae_xgb:>14.2f} {mae_ens:>14.2f}")
print("=" * 72)

# -------------------------------------------------------
# 2.4.2  Error histogram — signed error ŷ − y per engine
# -------------------------------------------------------
err_rf  = rul_pred_rf  - rul_true   # signed error (+ = overestimate)
err_xgb = rul_pred_xgb - rul_true

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bins = np.arange(-RUL_CAP, RUL_CAP + 10, 10)
for ax, err, label, rmse in zip(
        axes,
        [err_rf, err_xgb],
        [f'Random Forest (w={best_w_rf})', f'XGBoost (w={best_w_xgb})'],
        [rmse_test_rf, rmse_test_xgb]):
    ax.hist(err, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(0,  color='red',    linestyle='--', lw=1.2, label='No error')
    ax.axvline( rmse, color='orange', linestyle=':', lw=1.2, label=f'+RMSE ({rmse:.1f})')
    ax.axvline(-rmse, color='orange', linestyle=':', lw=1.2)
    ax.set_xlabel('Prediction error  ŷ − y  (cycles)')
    ax.set_ylabel('Number of engines')
    ax.set_title(f'{label}\nRMSE = {rmse:.1f} cycles,  MAE = {np.mean(np.abs(err)):.1f} cycles')
    ax.legend(fontsize=8)
plt.suptitle('Test-set error distribution (100 engines)', fontsize=11)
plt.tight_layout()
plt.savefig('Error_Histogram.png', dpi=150, bbox_inches='tight')
plt.show(block=False)

# -------------------------------------------------------
# 2.4.3  Absolute error vs true RUL — where errors are largest
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, err, label in zip(
        axes,
        [err_rf, err_xgb],
        [f'Random Forest (w={best_w_rf})', f'XGBoost (w={best_w_xgb})']):
    ax.scatter(rul_true, np.abs(err), alpha=0.7, edgecolors='k', linewidths=0.3, s=40)
    ax.axhline(20, color='red', linestyle='--', lw=1.2, label='20-cycle target')
    ax.set_xlabel('True RUL (cycles)')
    ax.set_ylabel('|Prediction error| (cycles)')
    ax.set_title(label)
    ax.set_xlim(0, RUL_CAP)
    ax.legend(fontsize=8)
plt.suptitle('Absolute error vs true RUL — test engines', fontsize=11)
plt.tight_layout()
plt.savefig('Error_vs_RUL.png', dpi=150, bbox_inches='tight')
plt.show(block=True)

print("\nDone. Output files saved:")
print("  Feature_Variability.png  Sensor_Trends.png  Correlation_Heatmap.png")
print("  RUL_Validation.png  Feature_Importance.png")
print("  RUL_Test_Predictions.png  Error_Histogram.png  Error_vs_RUL.png")
