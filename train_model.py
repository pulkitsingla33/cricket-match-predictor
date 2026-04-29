import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

def prepare_player_features(batting_df, bowling_df, matches_df):
    """
    Calculate pre-match player averages to avoid data leakage.
    Uses expanding mean shifted by 1 match.
    """
    print("Engineering player pre-match strength features...")
    
    # Sort matches by date to ensure proper cumulative calculation
    matches_meta = matches_df[['match_id', 'date']].copy()
    matches_meta['date'] = pd.to_datetime(matches_meta['date'])
    
    # -- Batting Features --
    batting = batting_df.merge(matches_meta, on='match_id').sort_values('date')
    # Calculate career strike rate BEFORE this match
    batting['pre_match_sr'] = batting.groupby('batter')['strike_rate'].transform(lambda x: x.expanding().mean().shift(1))
    # Calculate career batting average (runs per innings) BEFORE this match
    batting['pre_match_avg'] = batting.groupby('batter')['runs'].transform(lambda x: x.expanding().mean().shift(1))
    # Replace NaN with a reasonable default (average SR / avg)
    avg_sr = batting['strike_rate'].mean()
    batting['pre_match_sr'] = batting['pre_match_sr'].fillna(avg_sr)
    avg_runs = batting['runs'].mean()
    batting['pre_match_avg'] = batting['pre_match_avg'].fillna(avg_runs)
    
    # Aggregate to match-team level
    match_batting = batting.groupby(['match_id', 'batting_team']).agg(
        team_batting_sr=('pre_match_sr', 'mean'),
        team_batting_avg=('pre_match_avg', 'mean'),
    ).reset_index()
    match_batting = match_batting.rename(columns={'batting_team': 'team'})
    
    # -- Bowling Features --
    bowling = bowling_df.merge(matches_meta, on='match_id').sort_values('date')
    # Calculate career economy BEFORE this match
    bowling['pre_match_econ'] = bowling.groupby('bowler')['economy'].transform(lambda x: x.expanding().mean().shift(1))
    # Calculate bowling strike rate (wickets per match) BEFORE this match
    bowling['pre_match_bowl_wkts'] = bowling.groupby('bowler')['wickets'].transform(lambda x: x.expanding().mean().shift(1))
    avg_econ = bowling['economy'].mean()
    bowling['pre_match_econ'] = bowling['pre_match_econ'].fillna(avg_econ)
    avg_wkts = bowling['wickets'].mean()
    bowling['pre_match_bowl_wkts'] = bowling['pre_match_bowl_wkts'].fillna(avg_wkts)
    
    # Aggregate to match-team level
    match_bowling = bowling.groupby(['match_id', 'bowling_team']).agg(
        team_bowling_econ=('pre_match_econ', 'mean'),
        team_bowling_wkts=('pre_match_bowl_wkts', 'mean'),
    ).reset_index()
    match_bowling = match_bowling.rename(columns={'bowling_team': 'team'})
    
    return match_batting, match_bowling

def train_ensemble_model():
    training_start = time.perf_counter()
    # 1. Load Data
    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    batting_df = pd.read_csv(data_path / "features_batting.csv")
    bowling_df = pd.read_csv(data_path / "features_bowling.csv")
    
    # 2. Filter matches and drop NaNs in target
    df = df.dropna(subset=['team1_won'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 3. Engineer Player Strength Features
    match_batting, match_bowling = prepare_player_features(batting_df, bowling_df, df)
    
    # Merge for Team 1
    df = df.merge(match_batting.rename(columns={'team': 'team1', 'team_batting_sr': 'team1_batting_sr', 'team_batting_avg': 'team1_batting_avg'}), on=['match_id', 'team1'], how='left')
    df = df.merge(match_bowling.rename(columns={'team': 'team1', 'team_bowling_econ': 'team1_bowling_econ', 'team_bowling_wkts': 'team1_bowling_wkts'}), on=['match_id', 'team1'], how='left')
    
    # Merge for Team 2
    df = df.merge(match_batting.rename(columns={'team': 'team2', 'team_batting_sr': 'team2_batting_sr', 'team_batting_avg': 'team2_batting_avg'}), on=['match_id', 'team2'], how='left')
    df = df.merge(match_bowling.rename(columns={'team': 'team2', 'team_bowling_econ': 'team2_bowling_econ', 'team_bowling_wkts': 'team2_bowling_wkts'}), on=['match_id', 'team2'], how='left')
    
    # 4. Compute differential features (team1 advantage over team2)
    df['form_diff'] = df['team1_recent_form'] - df['team2_recent_form']
    df['batting_sr_diff'] = df['team1_batting_sr'] - df['team2_batting_sr']
    df['batting_avg_diff'] = df['team1_batting_avg'] - df['team2_batting_avg']
    df['bowling_econ_diff'] = df['team2_bowling_econ'] - df['team1_bowling_econ']  # higher diff = team1 bowls better
    df['bowling_wkts_diff'] = df['team1_bowling_wkts'] - df['team2_bowling_wkts']
    df['venue_wr_diff'] = df['team1_venue_win_rate'] - df['team2_venue_win_rate']
    df['pp_runs_diff'] = df['team1_avg_pp_runs'] - df['team2_avg_pp_runs']
    df['death_wkts_diff'] = df['team1_avg_death_wkts'] - df['team2_avg_death_wkts']
    
    # 5. Define Features
    # Removed 'venue' from categorical — its signal is captured by venue win rate features
    cat_features = ['team1', 'team2']
    num_features = [
        # Toss
        'toss_winner_is_team1', 'toss_bat_first', 
        # Historical rates
        'h2h_team1_win_rate', 'h2h_matches',
        'team1_venue_win_rate', 'team2_venue_win_rate',
        'team1_recent_form', 'team2_recent_form',
        # Player strength
        'team1_batting_sr', 'team2_batting_sr', 
        'team1_batting_avg', 'team2_batting_avg',
        'team1_bowling_econ', 'team2_bowling_econ',
        'team1_bowling_wkts', 'team2_bowling_wkts',
        # New features from feature engineering
        'team1_win_after_batting_first', 'team2_win_after_batting_first',
        'team1_avg_pp_runs', 'team2_avg_pp_runs',
        'team1_avg_death_wkts', 'team2_avg_death_wkts',
        'is_home_team1', 'is_home_team2',
        'season_progress',
        # Differential features
        'form_diff', 'batting_sr_diff', 'batting_avg_diff',
        'bowling_econ_diff', 'bowling_wkts_diff',
        'venue_wr_diff', 'pp_runs_diff', 'death_wkts_diff',
    ]
    
    X = df[cat_features + num_features]
    y = df['team1_won']
    
    # 6. Chronological Split
    # Training on matches before 2023, testing on 2023-2024
    train_idx = df['date'] < '2023-01-01'
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]
    
    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches")
    print(f"Feature count: {len(cat_features)} categorical + {len(num_features)} numerical = {len(cat_features) + len(num_features)} total")
    
    # 7. Build Pipeline
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
    
    # 8. Define Ensemble (Stacking) with tuned hyperparameters
    base_learners = [
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features='sqrt', random_state=42
        )),
        ('gb', HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, min_samples_leaf=15,
            learning_rate=0.05, random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            random_state=42
        ))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    
    bundle_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stacking_clf)
    ])
    
    # 9. Cross-validation on training set
    print("\nRunning 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(bundle_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # 10. Train on full training set
    print("\nTraining Stacking Ensemble Model on full training set...")
    bundle_pipeline.fit(X_train, y_train)
    
    # 11. Evaluate
    y_pred = bundle_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance on Test Set (2023-2024 Seasons):")
    print(f"Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 11.5 Generate ROC/AUC
    print("\nGenerating ROC/AUC graph...")
    y_prob = bundle_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Win Prediction')
    plt.legend(loc="lower right")
    
    plots_dir = data_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "win_model_roc.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to {plot_path}")
    
    # 12. Feature Importance Analysis
    print("\n" + "─"*55)
    print("FEATURE IMPORTANCE (from Random Forest base learner)")
    print("─"*55)
    # Get feature names after preprocessing
    preprocessor_fitted = bundle_pipeline.named_steps['preprocessor']
    num_names = num_features
    cat_names = list(preprocessor_fitted.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(cat_features))
    all_feature_names = num_names + cat_names
    
    # Get RF importances from the stacking classifier
    rf_model = bundle_pipeline.named_steps['classifier'].estimators_[0]
    importances = rf_model.feature_importances_
    
    # Sort and display top 20
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    print("\nTop 20 features:")
    for feat, imp in feat_imp.head(20).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<40s} {imp:.4f} {bar}")
    
    # 13. CV vs Test gap check (overfitting diagnostic)
    gap = abs(cv_scores.mean() - test_accuracy)
    print(f"\nCV-Test gap: {gap:.4f} {'✓ Good' if gap < 0.05 else '⚠ Possible overfitting'}")
    
    # 14. Save Model
    joblib.dump(bundle_pipeline, "ipl_ensemble_predictor.joblib")
    print(f"\nModel saved to ipl_ensemble_predictor.joblib")

    training_end = time.perf_counter()
    total_seconds = training_end - training_start
    print(f"Training completed in {total_seconds:.2f} seconds.")

if __name__ == "__main__":
    train_ensemble_model()
