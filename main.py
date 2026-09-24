import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Создаем папку для сохранения графиков
os.makedirs('images', exist_ok=True)

print("=" * 70)
print(" ЗАПУСК АНАЛИЗА ДАННЫХ: BREAST CANCER WISCONSIN")
print("=" * 70)

# ==============================================================================
# 1. ETL: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==============================================================================
print("\n[1/5] Загрузка и очистка данных (ETL)...")
df = pd.read_csv('data.csv')

# Исключаем малоинформативные поля (как в примере из методички)
cols_to_drop = [col for col in df.columns if 'Unnamed' in col or col == 'id']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Проверка пропусков и дубликатов
print(f"  - Пропуски: {df.isnull().sum().sum()}")
print(f"  - Дубликаты: {df.duplicated().sum()}")

# Кодирование целевой переменной (M = 1 (злокачественная), B = 0 (доброкачественная))
df['diagnosis_num'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("  ✅ Очистка завершена.")

# ==============================================================================
# 2. EDA: ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ
# ==============================================================================
print("\n[2/5] Проведение исследовательского анализа (EDA)...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'diagnosis_num' in numeric_cols:
    numeric_cols.remove('diagnosis_num')

# График 1: Распределение диагнозов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['diagnosis'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
plt.title('Распределение диагнозов', fontweight='bold')
plt.ylabel('')

# График 2: Топ-5 корреляций с диагнозом
plt.subplot(1, 2, 2)
corr = df[numeric_cols].corrwith(df['diagnosis_num']).abs().sort_values(ascending=False).head(5)
sns.barplot(x=corr.values, y=corr.index, palette='viridis')
plt.title('Топ-5 признаков, коррелирующих с диагнозом', fontweight='bold')
plt.xlabel('Абсолютная корреляция')

plt.tight_layout()
plt.savefig('images/01_eda_distribution.png', dpi=300)
plt.close()
print("  ✅ Графики EDA сохранены в папку 'images'.")

# ==============================================================================
# 3. ОБОГАЩЕНИЕ ДАННЫХ: КЛАСТЕРИЗАЦИЯ (Аналог ABC/RFM для мед. данных)
# ==============================================================================
print("\n[3/5] Обогащение данных: Кластеризация пациентов (K-Means)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# Применяем K-Means (k=3, как оптимальное для разделения на группы риска)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("  Распределение по кластерам:")
print(df['cluster'].value_counts().to_frame(name='Количество пациентов'))

# График 3: Связь кластеров с диагнозом
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='cluster', hue='diagnosis', palette=['#2ecc71', '#e74c3c'])
plt.title('Распределение реальных диагнозов по кластерам', fontweight='bold')
plt.xlabel('Кластер')
plt.ylabel('Количество пациентов')
plt.legend(title='Диагноз')
plt.savefig('images/02_clusters_vs_diagnosis.png', dpi=300)
plt.close()
print("  ✅ Кластеризация завершена.")

# ==============================================================================
# 4. МАШИННОЕ ОБУЧЕНИЕ: КЛАССИФИКАЦИЯ
# ==============================================================================
print("\n[4/5] Обучение и сравнение моделей машинного обучения...")
X = df[numeric_cols]
y = df['diagnosis_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    results.append({'Model': name, 'Accuracy': acc, 'ROC-AUC': roc})
    print(f"  - {name:20} | Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
best_model_name = results_df.iloc[0]['Model']
print(f"  🏆 Лучшая модель: {best_model_name}")

# Добавляем предсказания лучшей модели в исходный датасет для BI
# Используем лучшую модель из сравнения (Gradient Boosting)
best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
best_model.fit(X, y)

# Исправление: преобразуем numpy-массив в pd.Series перед .map()
predictions = best_model.predict(X)
df['predicted_diagnosis'] = pd.Series(predictions).map({0: 'B', 1: 'M'})

# Вероятности (работают без изменений, так как pandas сам преобразует массив)
df['prediction_probability'] = best_model.predict_proba(X)[:, 1]
# ==============================================================================
# 5. ЭКСПОРТ ДАННЫХ ДЛЯ BI-ПЛАТФОРМЫ
# ==============================================================================
print("\n[5/5] Подготовка и экспорт данных для Yandex DataLens...")
# Создаем категории для удобной визуализации в BI (аналог сегментации)
df['tumor_size'] = pd.qcut(df['area_mean'], q=3, labels=['Small', 'Medium', 'Large'])
df['severity'] = pd.qcut(df['concave points_mean'], q=2, labels=['Low', 'High'])

output_file = 'breast_cancer_ready_for_bi.csv'
df.to_csv(output_file, index=False)
print(f"  ✅ Файл '{output_file}' успешно создан в корне проекта!")
print("=" * 70)
print(" АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
print("=" * 70)