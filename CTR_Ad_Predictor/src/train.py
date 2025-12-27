import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier, Pool
from data_loader import load_and_preprocess_data


def make_data():
    X, y = load_and_preprocess_data()
    cat_features = list(X.columns)

    # Модели иногда сложно определить сложные зависимости, поэтому я ей немного подскажу, где искать
    X['product_campaign'] = X['product'].astype(str) + '_' + X['campaign_id'].astype(str)
    X['gender_age'] = X['gender'].astype(str) + '_' + X['age_level'].astype(str)
    X['product_webpage'] = X['product'].astype(str) + '_' + X['webpage_id'].astype(str)
    X['campaign_webpage'] = X['campaign_id'].astype(str) + '_' + X['webpage_id'].astype(str)

    cat_features += ['is_night', 'is_morning', 'is_day', 'product_campaign', 'gender_age',
                     'product_webpage', 'campaign_webpage']

    for col in X.columns:
        X[col] = X[col].astype(str)

    # Взял информацию из графиков EDA
    X['is_night'] = X['hour'].astype(int).isin([0, 1]).astype(int)
    X['is_morning'] = X['hour'].astype(int).isin([6, 7, 8, 9]).astype(int)
    X['is_day'] = X['weekday'].astype(int).isin([0, 1, 6]).astype(int)

    return X, y, cat_features


def train_sklearn():
    X, y, cat_features = make_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=10,
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=100,
        class_weights=[1, len(y) / y.sum() * 0.5]
    )

    model.fit(train_pool, eval_set=val_pool)

    preds_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds_proba)
    ll = log_loss(y_val, preds_proba)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"LogLoss: {ll:.4f}")

    model.save_model('../models/catboost_baseline.cbm')
    print("\nМодель сохранена в models/catboost_baseline.cbm")

    importance = model.get_feature_importance()
    feat_imp = pd.Series(importance, index=cat_features).sort_values(ascending=False)
    print("\nТоп 10 важных признаков:")
    print(feat_imp.head(10))


if __name__ == "__main__":
    train_sklearn()
