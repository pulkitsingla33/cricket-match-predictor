"""
IPL First Innings Score Predictor — Training Script
====================================================
Trains a regression model to predict first-innings total runs.

Features are all pre-match (no data leakage):
  - batting team, bowling team, venue  (categorical)
  - team batting/bowling strength      (from player CSVs)
  - historical averages                (from feature CSV)
  - toss/context info

Output: ipl_score_predictor.joblib
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


def prepare_player_features(batting_df, bowling_df, matches_df):
    """
    Calculate pre-match player averages to avoid data leakage.
    Uses expanding mean shifted by 1 match.
    (Same logic as train_model.py but returns batting/bowling team-level stats.)
    """
    print("Engineering player pre-match strength features...")

    matches_meta = matches_df[['match_id', 'date']].copy()
    matches_meta['date'] = pd.to_datetime(matches_meta['date'])

    # -- Batting Features --
    batting = batting_df.merge(matches_meta, on='match_id').sort_values('date')
    batting['pre_match_sr'] = batting.groupby('batter')['strike_rate'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    batting['pre_match_avg'] = batting.groupby('batter')['runs'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    avg_sr = batting['strike_rate'].mean()
    batting['pre_match_sr'] = batting['pre_match_sr'].fillna(avg_sr)
    avg_runs = batting['runs'].mean()
    batting['pre_match_avg'] = batting['pre_match_avg'].fillna(avg_runs)

    match_batting = batting.groupby(['match_id', 'batting_team']).agg(
        team_batting_sr=('pre_match_sr', 'mean'),
        team_batting_avg=('pre_match_avg', 'mean'),
    ).reset_index()
    match_batting = match_batting.rename(columns={'batting_team': 'team'})

    # -- Bowling Features --
    bowling = bowling_df.merge(matches_meta, on='match_id').sort_values('date')
    bowling['pre_match_econ'] = bowling.groupby('bowler')['economy'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    bowling['pre_match_bowl_wkts'] = bowling.groupby('bowler')['wickets'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    avg_econ = bowling['economy'].mean()
    bowling['pre_match_econ'] = bowling['pre_match_econ'].fillna(avg_econ)
    avg_wkts = bowling['wickets'].mean()
    bowling['pre_match_bowl_wkts'] = bowling['pre_match_bowl_wkts'].fillna(avg_wkts)

    match_bowling = bowling.groupby(['match_id', 'bowling_team']).agg(
        team_bowling_econ=('pre_match_econ', 'mean'),
        team_bowling_wkts=('pre_match_bowl_wkts', 'mean'),
    ).reset_index()
    match_bowling = match_bowling.rename(columns={'bowling_team': 'team'})

    return match_batting, match_bowling


def train_score_model():
    training_start = time.perf_counter()
    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    batting_df = pd.read_csv(data_path / "features_batting.csv")
    bowling_df = pd.read_csv(data_path / "features_bowling.csv")

    # Filter: need valid innings1_runs
    df = df.dropna(subset=['innings1_runs'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Player strength features
    match_batting, match_bowling = prepare_player_features(batting_df, bowling_df, df)

    # Merge for innings1_team (batting team) 
    df = df.merge(
        match_batting.rename(columns={
            'team': 'innings1_team',
            'team_batting_sr': 'bat_team_sr',
            'team_batting_avg': 'bat_team_avg',
        }),
        on=['match_id', 'innings1_team'], how='left'
    )
    df = df.merge(
        match_bowling.rename(columns={
            'team': 'innings1_team',
            'team_bowling_econ': 'bat_team_bowl_econ',
            'team_bowling_wkts': 'bat_team_bowl_wkts',
        }),
        on=['match_id', 'innings1_team'], how='left'
    )

    # Merge for innings2_team (bowling team in first innings)
    df = df.merge(
        match_batting.rename(columns={
            'team': 'innings2_team',
            'team_batting_sr': 'bowl_team_bat_sr',
            'team_batting_avg': 'bowl_team_bat_avg',
        }),
        on=['match_id', 'innings2_team'], how='left'
    )
    df = df.merge(
        match_bowling.rename(columns={
            'team': 'innings2_team',
            'team_bowling_econ': 'bowl_team_econ',
            'team_bowling_wkts': 'bowl_team_wkts',
        }),
        on=['match_id', 'innings2_team'], how='left'
    )

    # ── Features ──────────────────────────────────────────
    cat_features = ['innings1_team', 'innings2_team', 'venue']
    num_features = [
        # Batting team strength
        'bat_team_sr', 'bat_team_avg',
        # Bowling team (fielding first) strength
        'bowl_team_econ', 'bowl_team_wkts',
        # Historical rolling averages
        'team1_avg_pp_runs', 'team2_avg_pp_runs',
        'team1_avg_death_wkts', 'team2_avg_death_wkts',
        'team1_win_after_batting_first', 'team2_win_after_batting_first',
        # Context
        'toss_bat_first',
        'is_home_team1', 'is_home_team2',
        'season_progress',
        # Era / rule-change flag (IPL 2023+: 12-player squads, higher scoring)
        'is_impact_player_era',
    ]

    X = df[cat_features + num_features]
    y = df['innings1_runs']

    # Chronological split
    train_idx = df['date'] < '2023-01-01'
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]

    print(f"\nTraining on {len(X_train)} matches, testing on {len(X_test)} matches")
    print(f"Feature count: {len(cat_features)} categorical + {len(num_features)} numerical = {len(cat_features) + len(num_features)} total")

    # ── Pipeline ──────────────────────────────────────────
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    # Stacking Regressor
    base_learners = [
        ('rf', RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=8,
            max_features='sqrt', random_state=42
        )),
        ('gb', HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, min_samples_leaf=12,
            learning_rate=0.05, random_state=42
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=8,
            random_state=42
        )),
    ]

    stacking_reg = StackingRegressor(
        estimators=base_learners,
        final_estimator=Ridge(alpha=1.0),
        cv=5
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', stacking_reg)
    ])

    # Cross-validation
    print("\nRunning 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    print(f"CV MAE: {cv_mae:.2f} ± {cv_scores.std():.2f}")

    # Train
    print("\nTraining Stacking Ensemble Regressor on full training set...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    print("\n" + "-" * 55)
    print("SCORE PREDICTION MODEL - Test Set (2023-2024 Seasons)")
    print("-" * 55)
    print(f"  MAE  : {test_mae:.2f} runs")
    print(f"  RMSE : {test_rmse:.2f} runs")
    print(f"  R2   : {test_r2:.4f}")
    
    # ── Plotting ──────────────────────────────────────────
    print("\nGenerating Regression Diagnostic Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='teal')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[0].set_xlabel('Actual Runs')
    axes[0].set_ylabel('Predicted Runs')
    axes[0].set_title('Predicted vs Actual Runs')
    
    # 2. Residual Plot
    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='crimson')
    axes[1].axhline(y=0, color='k', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Runs')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title('Residuals vs Predicted')
    
    plt.tight_layout()
    plots_dir = data_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "score_model_regression.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Regression plots saved to {plot_path}")

    # Feature importance (from Random Forest)
    print("\n" + "-" * 55)
    print("FEATURE IMPORTANCE (from Random Forest base learner)")
    print("-" * 55)
    preprocessor_fitted = pipeline.named_steps['preprocessor']
    num_names = num_features
    cat_names = list(preprocessor_fitted.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(cat_features))
    all_feature_names = num_names + cat_names

    rf_model = pipeline.named_steps['regressor'].estimators_[0]
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    print("\nTop 15 features:")
    for feat, imp in feat_imp.head(15).items():
        bar = "#" * int(imp * 200)
        print(f"  {feat:<45s} {imp:.4f} {bar}")

    # Residual analysis
    residuals = y_test.values - y_pred
    print(f"\n  Mean residual  : {residuals.mean():+.2f} runs")
    print(f"  Residual std   : {residuals.std():.2f} runs")
    print(f"  Within ±15 runs: {(np.abs(residuals) <= 15).mean()*100:.1f}%")
    print(f"  Within ±25 runs: {(np.abs(residuals) <= 25).mean()*100:.1f}%")

    # Save
    model_path = data_path / "ipl_score_predictor.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path.name}")

    # Save metadata including era-aware residual stats for confidence intervals
    # Since chronological split puts all 2023+ (impact-era) matches in test set,
    # we store the mean bias (systematic under/over-prediction) for correction in the TUI.
    test_era = df.loc[~train_idx, 'is_impact_player_era'].values
    residuals_post = (y_test.values - y_pred)[test_era == 1] if (test_era == 1).any() else residuals
    residuals_pre  = (y_test.values - y_pred)[test_era == 0] if (test_era == 0).any() else np.array([])

    metadata = {
        'residual_std': float(residuals.std()),
        'residual_std_impact_era': float(residuals_post.std()) if len(residuals_post) > 0 else float(residuals.std()),
        'residual_std_pre_impact': float(residuals_pre.std()) if len(residuals_pre) > 0 else float(residuals.std()),
        # Mean bias: positive = model underpredicts; used in TUI to correct predictions for impact era
        'impact_era_bias': float(residuals_post.mean()) if len(residuals_post) > 0 else 0.0,
        'test_mae': float(test_mae),
        'cv_mae': float(cv_mae),
        'impact_era_avg_score': float(df[df['is_impact_player_era'] == 1]['innings1_runs'].mean()),
        'pre_impact_avg_score': float(df[df['is_impact_player_era'] == 0]['innings1_runs'].mean()),
    }
    print(f"\n  Impact era bias correction: {metadata['impact_era_bias']:+.2f} runs")
    joblib.dump(metadata, data_path / "ipl_score_predictor_meta.joblib")
    print(f"Metadata saved to ipl_score_predictor_meta.joblib")

    training_end = time.perf_counter()
    total_seconds = training_end - training_start
    print(f"Training completed in {total_seconds:.2f} seconds.")

if __name__ == "__main__":
    train_score_model()
