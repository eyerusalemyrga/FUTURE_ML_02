import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def find_path(filename_segment):
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        for file in files:
            if filename_segment in file:
                return os.path.join(root, file)
    return None

def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        try: return pd.read_csv(path)
        except UnicodeDecodeError: return pd.read_csv(path, encoding='latin1')


file_targets = {
    "Telecom": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "Banking": "Churn_Modelling.csv",
    "Music (Spotify)": "Spotify_data"
}

for name, file_name in file_targets.items():
    path = find_path(file_name)
    if not path:
        print(f"Skipping {name}: File not found.")
        continue

    print(f"\n{'='*40}\nPROCESING: {name}\n{'='*40}")
    df = load_data(path)
    df.head()

    if name == "Telecom":
        target = 'Churn'
        df.drop(columns=['customerID'], errors='ignore', inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    elif name == "Banking":
        target = 'Exited'
        df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore', inplace=True)
        num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    elif name == "Music (Spotify)":
        target = 'premium_sub_willingness'
        df[target] = df[target].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        num_cols = ['music_recc_rating']


    print(f"\n--- {name} Summary Statistics ---")
    stats = df[num_cols].agg(['mean', 'median']).transpose()
    print(stats)

    plt.figure(figsize=(15, 4))
    for i, col in enumerate(num_cols):
        plt.subplot(1, len(num_cols), i+1)
        sns.histplot(df[col], kde=True, color='forestgreen')
        plt.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean')
        plt.axvline(df[col].median(), color='orange', linestyle='-', label=f'Median')
        plt.title(f'{col} Distribution')
        plt.legend()
    plt.tight_layout()
    plt.show()

    le = LabelEncoder()
    df_ml = df.copy()
    for col in df_ml.columns:
        if df_ml[col].dtype == 'object':
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    X = df_ml.drop(target, axis=1)
    y = df_ml[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)


    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()

    
