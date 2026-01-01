# %%
# 1. Import Libraries

import pandas as pd 

from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak ayırmak için gerekli fonksiyon

from sklearn.naive_bayes import GaussianNB  # Sürekli sayısal veriler için Naive Bayes sınıflandırıcısı

from sklearn.naive_bayes import BernoulliNB  # Binary (0/1) veriler için Naive Bayes sınıflandırıcısı

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
# Model performansını ölçmek için; doğruluk, karışıklık matrisi ve detaylı sınıflandırma raporu

from sklearn.metrics import roc_auc_score, roc_curve
# ROC AUC skoru ve ROC eğrisi için gerekli metrikler

from sklearn.tree import DecisionTreeClassifier  # Karar ağacı sınıflandırıcısı

from sklearn.ensemble import RandomForestClassifier  # Rastgele orman (birden fazla karar ağacı) sınıflandırıcısı

from sklearn.neighbors import KNeighborsClassifier  # K-en yakın komşu algoritması sınıflandırıcısı

from sklearn.ensemble import GradientBoostingClassifier  # Gradyan artırımlı karar ağacı sınıflandırıcısı

from sklearn.linear_model import LogisticRegression  # Lojistik regresyon sınıflandırıcısı (0/1 tahmini)

# %%
# 2. Load Clean Data

df = pd.read_csv('../data/clean_train.csv')

df.head()

# 3. Modeling
# %%
# x, y ayrımı

x = df.drop('Credit_Score', axis=1)  # Özellikler (bağımsız değişkenler)
y = df['Credit_Score']  # Hedef değişken (bağımlı değişken)

# %%
# get_dummies ile kategorik değişkenleri sayısal hale getirme

x = pd.get_dummies(x, drop_first=True)

x.head()        

# %%
# Veriyi eğitim ve test olarak ayırma

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)  

# Veriyi eğitim ve test olarak ayırıyoruz
# x -> özellikler (features), y -> hedef değişken (target)
# test_size=0.15 -> verinin %15'i test için ayrılır, %85'i eğitim için
# random_state=42 -> veriyi her çalıştırmada aynı şekilde bölmek için sabit sayı

# %%
# Lojistik Regresyon Modeli

L = LogisticRegression()  
# Lojistik Regresyon modelini oluşturuyoruz (sınıflandırma için, 0/1 tahmini)

L.fit(x_train, y_train)  
# Modeli eğitim verisi ile eğitiyoruz

Ltahmin = L.predict(x_test)  
# Test verisi üzerinde tahmin yapıyoruz

accuracy_score(y_test, Ltahmin)  
# Modelin doğruluk (accuracy) skorunu hesaplıyoruz

confusion_matrix(y_test, Ltahmin)  
# Karışıklık matrisini hesaplıyoruz (gerçek vs tahmin değerleri)

print(classification_report(y_test, Ltahmin))  
# Detaylı sınıflandırma raporu: precision, recall, f1-score ve support

# %%
# Algo Test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def algo_test(x, y):
    """
    Farklı sınıflandırma algoritmalarını test eder ve sonuçları döndürür.
    Multiclass hedef değişkeni destekler ve F1 skorlarına göre sıralama yapar.
    """
    # Modeller
    models = [
        GaussianNB(),
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        KNeighborsClassifier(),
        AdaBoostClassifier(random_state=42)
    ]

    names = [
        "GaussianNB",
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "KNN",
        "AdaBoost"
    ]

    # Veri setini ayır
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []

    print("🚀 Modeller eğitiliyor...\n")

    for model, name in zip(models, names):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Çok sınıflı uyumlu skorlar
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

        print(f"🔹 {name} Sonuçları:")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
        print("-" * 50)

    # DataFrame
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="F1", ascending=False, inplace=True)

    # En iyi model
    best_model_name = results_df.iloc[0]["Model"]
    print("\n🏆 En başarılı model:", best_model_name)

    return results_df

# Algo Test'i çalıştır ve sonuçları kaydet
results_df = algo_test(x, y)

# xlsx olarak kaydet
results_df.to_excel("model_results.xlsx", index=False)
print("📂 Model sonuçları kaydedildi: model_results.xlsx")

# %%
# RandomForestClassifier

import joblib

best_model = RandomForestClassifier(random_state=42)

best_model.fit(x, y)  # tüm veri ile yeniden eğit

# Modeli kaydet
joblib.dump(best_model, "random_forest_model.pkl")
print("📂 En iyi model kaydedildi: random_forest_model.pkl")


# Eğitim sırasında kullanılan feature kolonlarını kaydet
joblib.dump(x.columns.tolist(), "columns.pkl")








