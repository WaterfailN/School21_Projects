import pandas as pd

def load_and_preprocess_data(csv_path: str = "../data/Ad_Click_prediction_train.csv"):
    """
    Загружает и предобрабатывает данные для задачи предсказания CTR.

    Args:
        csv_path (str): Путь к CSV-файлу

    Возвращает:
        X (pd.DataFrame): Признаки
        y (pd.Series): Целевая переменная (0/1)
        cat_features (list[str]): Список названий категориальных колонок
    """

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['is_click']) # Если вдруг появятся пропуски, строки будут удалены

    # Здесь слишком много пропусков, решил удалить
    df = df.drop(columns=['product_category_2', 'city_development_index'])

    # Обработка времени
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['weekday'] = df['DateTime'].dt.weekday
    df = df.drop(columns=['DateTime', 'session_id', 'user_id'])

    X = df.drop(columns=['is_click'])
    y = df['is_click'].astype(int)

    for col in X.columns:
        X[col] = X[col].fillna(X[col].mode()[0])
    print("Данные очищены от пропусков:")
    print(X.isnull().sum())

    return X, y


if __name__ == "__main__":
    load_and_preprocess_data()