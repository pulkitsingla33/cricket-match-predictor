import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
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
    # Replace NaN with a reasonable default (average SR)
    avg_sr = batting['strike_rate'].mean()
    batting['pre_match_sr'] = batting['pre_match_sr'].fillna(avg_sr)
    
    # Aggregate to match-team level
    match_batting = batting.groupby(['match_id', 'batting_team'])['pre_match_sr'].mean().reset_index()
    match_batting = match_batting.rename(columns={'pre_match_sr': 'team_batting_sr', 'batting_team': 'team'})
    
    # -- Bowling Features --
    bowling = bowling_df.merge(matches_meta, on='match_id').sort_values('date')
    # Calculate career economy BEFORE this match
    bowling['pre_match_econ'] = bowling.groupby('bowler')['economy'].transform(lambda x: x.expanding().mean().shift(1))
    avg_econ = bowling['economy'].mean()
    bowling['pre_match_econ'] = bowling['pre_match_econ'].fillna(avg_econ)
    
    # Aggregate to match-team level
    match_bowling = bowling.groupby(['match_id', 'bowling_team'])['pre_match_econ'].mean().reset_index()
    match_bowling = match_bowling.rename(columns={'pre_match_econ': 'team_bowling_econ', 'bowling_team': 'team'})
    
    return match_batting, match_bowling

def train_ensemble_model():
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
    df = df.merge(match_batting.rename(columns={'team': 'team1', 'team_batting_sr': 'team1_batting_sr'}), on=['match_id', 'team1'], how='left')
    df = df.merge(match_bowling.rename(columns={'team': 'team1', 'team_bowling_econ': 'team1_bowling_econ'}), on=['match_id', 'team1'], how='left')
    
    # Merge for Team 2
    df = df.merge(match_batting.rename(columns={'team': 'team2', 'team_batting_sr': 'team2_batting_sr'}), on=['match_id', 'team2'], how='left')
    df = df.merge(match_bowling.rename(columns={'team': 'team2', 'team_bowling_econ': 'team2_bowling_econ'}), on=['match_id', 'team2'], how='left')
    
    # 4. Define Features
    cat_features = ['team1', 'team2', 'venue']
    num_features = [
        'toss_winner_is_team1', 'toss_bat_first', 
        'h2h_team1_win_rate', 'team1_venue_win_rate', 'team2_venue_win_rate',
        'team1_recent_form', 'team2_recent_form',
        'team1_batting_sr', 'team2_batting_sr', 
        'team1_bowling_econ', 'team2_bowling_econ'
    ]
    
    X = df[cat_features + num_features]
    y = df['team1_won']
    
    # 5. Chronological Split
    # Training on matches before 2023, testing on 2023-2024
    train_idx = df['date'] < '2023-01-01'
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]
    
    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches")
    
    # 6. Build Pipeline
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
    
    # 7. Define Ensemble (Stacking)
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', HistGradientBoostingClassifier(random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    bundle_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stacking_clf)
    ])
    
    # 8. Train
    print("Training Stacking Ensemble Model...")
    bundle_pipeline.fit(X_train, y_train)
    
    # 9. Evaluate
    y_pred = bundle_pipeline.predict(X_test)
    print("\nModel Performance on Test Set (2023-2024 Seasons):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 10. Save Model
    joblib.dump(bundle_pipeline, "ipl_ensemble_predictor.joblib")
    print(f"\nModel saved to ipl_ensemble_predictor.joblib")

if __name__ == "__main__":
    train_ensemble_model()
