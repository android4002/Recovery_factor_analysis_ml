# ======================================================================================
# ФАКТОРНЫЙ АНАЛИЗ ВОССТАНОВЛЕНИЯ ПОСЛЕ ИНФАРКТА МИОКАРДА
# ======================================================================================
# Мультиклассовая классификация (3 класса)
# SMOTE балансировка с k_neighbors=2
# RandomForest + XGBoost с GridSearch
# SHAP интерпретация
# Radar Chart для профилей классов
# ======================================================================================

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, accuracy_score)
# ML библиотеки
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Настройки
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# Создание директории для результатов
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'RESULT_ANALYSIS_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("ФАКТОРНЫЙ АНАЛИЗ ВОССТАНОВЛЕНИЯ ПОСЛЕ ИНФАРКТА МИОКАРДА")
print("=" * 80)
print(f" Создана папка: {output_dir}\n")

# ======================================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ======================================================================================
print("=" * 80)
print("1. ЗАГРУЗКА И ПЕРВИЧНЫЙ ПРОСМОТР ДАННЫХ")
print("=" * 80)

# Загрузка файла
file_path = '/Users/android/Downloads/Для факт ан 25.01.26.xlsx'
df = pd.read_excel(file_path)

print(f"Загружено: {len(df)} пациентов, {df.shape[1]} признаков")
print(f"\nНазвания столбцов:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2}. {col}")

# ======================================================================================
# 2. ФОРМИРОВАНИЕ 3 КЛАССОВ ИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (Unnamed: 31)
# ======================================================================================
print("\n" + "=" * 80)
print("2. ФОРМИРОВАНИЕ 3 КЛАССОВ ИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("=" * 80)

# Находим целевую переменную (последний столбец или Unnamed: 31)
target_col = df.columns[-1]  # Последний столбец
print(f"Целевая переменная: '{target_col}'")

# Проверяем уникальные значения
print(f"\nУникальные значения ДО обработки:")
unique_vals = df[target_col].value_counts(dropna=False)
print(unique_vals)


# Функция парсинга целевой переменной
def parse_outcome(text):
    """
    Парсит текстовый исход в 3 класса:
    """
    if pd.isna(text): #обнаружение пропущенных
        return None

    text = str(text).lower().strip()

    # Class 0: Смерть
    if 'смерть' in text:
        return 0

    # Class 1: Не восстановился (но не смерть)
    elif 'не восст' in text:
        return 1

    # Class 2: Восстановился (содержит "восст" без "не")
    elif 'восст' in text and 'не' not in text:
        return 2

    else:
        return None  # Неизвестные значения (например, "выбыл")


# Применяем парсинг
df['Target'] = df[target_col].apply(parse_outcome)

# Удаляем строки с неопределенным исходом
df_clean = df[df['Target'].notna()].copy() #непропущенные данные, возвр True если все норм

print(f"\n{'=' * 80}")
print("РЕЗУЛЬТАТ ПАРСИНГА:")
print(f"{'=' * 80}")
class_mapping = {0: 'Смерть', 1: 'Не восстановился', 2: 'Восстановился'}
print("\nРаспределение классов:")
for cls, count in df_clean['Target'].value_counts().sort_index().items():
    print(f"  Class {int(cls)} ({class_mapping[int(cls)]:20}): {count:3} ({count / len(df_clean) * 100:5.1f}%)")

print(f"\n Удалено строк с неопределенным исходом: {len(df) - len(df_clean)}")
print(f" Итого пациентов для анализа: {len(df_clean)}")

# ======================================================================================
# 3. УДАЛЕНИЕ ПРИЗНАКОВ-УТЕЧЕК (из будущего)
# ======================================================================================
print("\n" + "=" * 80)
print("3. УДАЛЕНИЕ ПРИЗНАКОВ-УТЕЧЕК ДАННЫХ")
print("=" * 80)

# Список признаков, которые содержат информацию из будущего
leak_features = ['ФВ 1 год', 'ХСН1 год - 1', 'реаб1']

print("Признаки для удаления (содержат данные из будущего):")
leaked_found = []
for leak in leak_features:
    if leak in df_clean.columns:
        leaked_found.append(leak)
        print(f"   {leak} - УДАЛЕН")
    else:
        print(f"   {leak} - не найден в данных")

if leaked_found:
    df_clean = df_clean.drop(columns=leaked_found)

# Также удалим ФИО и исходный столбец с целевой переменной
non_feature_cols = ['ФИО пациента', target_col]
for col in non_feature_cols:
    if col in df_clean.columns:
        df_clean = df_clean.drop(columns=col)
        print(f"   {col} - удален (не признак)")

print(f"\n Осталось признаков: {df_clean.shape[1] - 1} (+ Target)")

# ======================================================================================
# 4. ПРЕДВАРИТЕЛЬНАЯ ОЧИСТКА ДАННЫХ
# ======================================================================================
print("\n" + "=" * 80)
print("4. ПРЕДВАРИТЕЛЬНАЯ ОЧИСТКА ДАННЫХ")
print("=" * 80)

# Сохраняем Target отдельно
y = df_clean['Target'].copy() # Целевая переменная
X = df_clean.drop(columns=['Target']) # Признаки (все, кроме целевой переменной)

print(f"Размерность до очистки: {X.shape}")

# Статистика пропусков
missing_stats = pd.DataFrame({
    'Признак': X.columns,
    'Пропуски': X.isnull().sum(),
    'Процент': (X.isnull().sum() / len(X) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Пропуски'] > 0].sort_values('Процент', ascending=False)

if len(missing_stats) > 0:
    print(f"\nПризнаки с пропусками (>{0}%):")
    print(missing_stats.to_string(index=False))

    # Удаляем признаки с >50% пропусков
    high_missing = missing_stats[missing_stats['Процент'] > 50]['Признак'].tolist()
    if high_missing:
        print(f"\n Удаляем признаки с >50% пропусков:")
        for feat in high_missing:
            print(f"  • {feat}")
        X = X.drop(columns=high_missing)

    # Для остальных: удаляем строки с пропусками
    rows_before = len(X)
    mask = X.isnull().any(axis=1)
    X = X[~mask]
    y = y[~mask]
    rows_removed = rows_before - len(X)

    if rows_removed > 0:
        print(f"\n Удалено строк с пропусками: {rows_removed}")
else:
    print(" Пропусков не обнаружено")

print(f"\nРазмерность после очистки: {X.shape}")
print(f"Пациентов: {len(X)}")

# Проверка баланса классов ПОСЛЕ очистки
print("\n" + "=" * 80)
print("БАЛАНС КЛАССОВ ПОСЛЕ ОЧИСТКИ:")
print("=" * 80)
class_counts_before = y.value_counts().sort_index()
for cls, count in class_counts_before.items():
    print(f"  Class {int(cls)} ({class_mapping[int(cls)]:20}): {count:3} ({count / len(y) * 100:5.1f}%)")

# ======================================================================================
# 5. ОПРЕДЕЛЕНИЕ ТИПОВ ПРИЗНАКОВ
# ======================================================================================
print("\n" + "=" * 80)
print("5. ОПРЕДЕЛЕНИЕ ТИПОВ ПРИЗНАКОВ")
print("=" * 80)

# Автоматическое определение числовых и категориальных признаков
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Уточним: если числовой признак имеет мало уникальных значений, это категориальный
for col in numerical_features.copy():
    if X[col].nunique() <= 10:  # Эвристика: <=10 уникальных значений = категориальный
        categorical_features.append(col)
        numerical_features.remove(col)

print(f"Числовые признаки ({len(numerical_features)}):")
for feat in numerical_features[:15]:  # Первые 15
    print(f"  • {feat}")
if len(numerical_features) > 15:
    print(f"  ... и еще {len(numerical_features) - 15}")

print(f"\nКатегориальные признаки ({len(categorical_features)}):")
for feat in categorical_features:
    print(f"  • {feat} ({X[feat].nunique()} категорий)")

# ======================================================================================
# 6. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ======================================================================================
print("\n" + "=" * 80)
print("6. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ (One-Hot)")
print("=" * 80)

if len(categorical_features) > 0:
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    print(f" Применено One-Hot кодирование")
    print(f"  Признаков ДО: {X.shape[1]}")
    print(f"  Признаков ПОСЛЕ: {X_encoded.shape[1]}")
else:
    X_encoded = X.copy()
    print(" Категориальных признаков не обнаружено")

# ======================================================================================
# 7. УДАЛЕНИЕ ВЫБРОСОВ (Isolation Forest)
# ======================================================================================
print("\n" + "=" * 80)
print("7. УДАЛЕНИЕ ВЫБРОСОВ (ISOLATION FOREST)")
print("=" * 80)

# Isolation Forest применяем только к числовым признакам
if len(numerical_features) > 0:
    iso_forest = IsolationForest(
        contamination=0.05,  # 5% выбросов
        random_state=42,
        n_estimators=200
    )

    outlier_labels = iso_forest.fit_predict(X_encoded[numerical_features])
    n_outliers = (outlier_labels == -1).sum()

    # Удаляем выбросы
    X_encoded = X_encoded[outlier_labels == 1]
    y = y[outlier_labels == 1]

    print(f" Удалено выбросов: {n_outliers} ({n_outliers / len(outlier_labels) * 100:.1f}%)")
    print(f" Осталось пациентов: {len(X_encoded)}")
else:
    print(" Числовых признаков недостаточно для Isolation Forest")

# Обновляем статистику классов
print("\n" + "=" * 80)
print("БАЛАНС КЛАССОВ ПОСЛЕ УДАЛЕНИЯ ВЫБРОСОВ:")
print("=" * 80)
class_counts_after_outliers = y.value_counts().sort_index()
for cls, count in class_counts_after_outliers.items():
    print(f"  Class {int(cls)} ({class_mapping[int(cls)]:20}): {count:3} ({count / len(y) * 100:5.1f}%)")

# ======================================================================================
# 8. СТАНДАРТИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ
# ======================================================================================
print("\n" + "=" * 80)
print("8. СТАНДАРТИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 80)

scaler = StandardScaler()
X_scaled = X_encoded.copy()

if len(numerical_features) > 0:
    X_scaled[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    print(f" Стандартизировано {len(numerical_features)} числовых признаков")
    print(f"  (μ=0, σ=1 для каждого признака)")
else:
    print(" Нет числовых признаков для стандартизации")

# ======================================================================================
# 9. БАЛАНСИРОВКА КЛАССОВ (SMOTE с k_neighbors=2)
# ======================================================================================
print("\n" + "=" * 80)
print("9. БАЛАНСИРОВКА КЛАССОВ (SMOTE)")
print("=" * 80)

# Проверяем минимальный класс
min_class_count = y.value_counts().min()
print(f"Минимальный размер класса: {min_class_count}")

# КРИТИЧЕСКИ ВАЖНО: k_neighbors должно быть меньше минимального класса
if min_class_count <= 1:
    print(" ОШИБКА: Класс 'Смерть' имеет ≤1 примера. SMOTE невозможен.")
    print("   Рекомендация: объедините классы или соберите больше данных.")
    k_neighbors_smote = None
elif min_class_count <= 5:
    k_neighbors_smote = 1  # Если класс очень мал (2-5 примеров)
    print(f" Малый класс ({min_class_count} примеров). Используем k_neighbors={k_neighbors_smote}")
else:
    k_neighbors_smote = 2  # По требованию задачи
    print(f" Используем k_neighbors={k_neighbors_smote}")

if k_neighbors_smote is not None:
    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors_smote,
        sampling_strategy='auto'  # Балансируем все классы до размера мажоритарного
    )

    print("\nПрименение SMOTE...")
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print(f"\n{'=' * 80}")
    print("РЕЗУЛЬТАТ SMOTE:")
    print(f"{'=' * 80}")
    print(f"ДО SMOTE:  {len(y)} примеров")
    print(f"ПОСЛЕ SMOTE: {len(y_resampled)} примеров (+{len(y_resampled) - len(y)} синтетических)\n")

    # Визуализация ДО и ПОСЛЕ
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ДО SMOTE
    class_counts_before_smote = y.value_counts().sort_index()
    colors_before = ['#e74c3c', '#f39c12', '#2ecc71']  # Красный, Оранжевый, Зеленый
    axes[0].bar(
        [class_mapping[int(cls)] for cls in class_counts_before_smote.index],
        class_counts_before_smote.values,
        color=colors_before,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    axes[0].set_title('ДО SMOTE (Несбалансированные классы)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Количество примеров', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for i, (cls, count) in enumerate(class_counts_before_smote.items()):
        axes[0].text(i, count + 1, str(count), ha='center', fontsize=12, fontweight='bold')

    # ПОСЛЕ SMOTE
    class_counts_after_smote = y_resampled.value_counts().sort_index()
    axes[1].bar(
        [class_mapping[int(cls)] for cls in class_counts_after_smote.index],
        class_counts_after_smote.values,
        color=colors_before,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    axes[1].set_title('ПОСЛЕ SMOTE (Сбалансированные классы)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Количество примеров', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for i, (cls, count) in enumerate(class_counts_after_smote.items()):
        axes[1].text(i, count + 1, str(count), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_SMOTE_балансировка.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Сохранен график: 01_SMOTE_балансировка.png\n")

    # Детализация по классам
    print("Распределение ПОСЛЕ SMOTE:")
    for cls, count in class_counts_after_smote.items():
        original_count = class_counts_before_smote.get(cls, 0)
        synthetic_count = count - original_count
        print(f"  Class {int(cls)} ({class_mapping[int(cls)]:20}): "
              f"{count:3} (оригинал: {original_count}, синтетика: {synthetic_count})")

else:
    # Если SMOTE невозможен, работаем с несбалансированными данными
    print("\n SMOTE не применен. Работаем с несбалансированными данными.")
    X_resampled = X_scaled
    y_resampled = y

# ======================================================================================
# 10. РАЗДЕЛЕНИЕ НА TRAIN/TEST
# ======================================================================================
print("\n" + "=" * 80)
print("10. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.25,  # 75% train, 25% test
    random_state=42,
    stratify=y_resampled  # Сохраняем пропорции классов
)

print(f" Train: {len(X_train)} примеров ({len(X_train) / len(X_resampled) * 100:.1f}%)")
print(f" Test:  {len(X_test)} примеров ({len(X_test) / len(X_resampled) * 100:.1f}%)")

print("\nРаспределение классов в Train:")
for cls, count in y_train.value_counts().sort_index().items():
    print(f"  Class {int(cls)} ({class_mapping[int(cls)]:20}): {count}")

print("\nРаспределение классов в Test:")
for cls, count in y_test.value_counts().sort_index().items():
    print(f"  Class {int(cls)} ({class_mapping[int(cls)]:20}): {count}")

# ======================================================================================
# СОХРАНЕНИЕ ПРОМЕЖУТОЧНЫХ РЕЗУЛЬТАТОВ
# ======================================================================================
print("\n" + "=" * 80)
print("СОХРАНЕНИЕ ПРОМЕЖУТОЧНЫХ ДАННЫХ")
print("=" * 80)

# Сохраняем обработанные данные
pd.DataFrame(X_train).to_csv(f'{output_dir}/X_train.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(X_test).to_csv(f'{output_dir}/X_test.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'Target': y_train}).to_csv(f'{output_dir}/y_train.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'Target': y_test}).to_csv(f'{output_dir}/y_test.csv', index=False, encoding='utf-8-sig')

print(" Сохранены:")
print("  • X_train.csv")
print("  • X_test.csv")
print("  • y_train.csv")
print("  • y_test.csv")

# Сохраняем mapping классов
import json

with open(f'{output_dir}/class_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(class_mapping, f, ensure_ascii=False, indent=2)
print("  • class_mapping.json")

# Статистика обработки
processing_stats = {
    'Исходное количество пациентов': len(df),
    'После парсинга целевой переменной': len(df_clean),
    'После удаления пропусков': len(X),
    'После удаления выбросов': len(X_scaled),
    'После SMOTE': len(X_resampled),
    'Train выборка': len(X_train),
    'Test выборка': len(X_test),
    'Количество признаков': X_train.shape[1],
    'Числовых признаков': len(numerical_features),
    'Категориальных признаков (до One-Hot)': len(categorical_features),
    'SMOTE k_neighbors': k_neighbors_smote if k_neighbors_smote else 'Не применен'
}

with open(f'{output_dir}/processing_stats.json', 'w', encoding='utf-8') as f:
    json.dump(processing_stats, f, ensure_ascii=False, indent=2)
print("  • processing_stats.json")

print("\n" + "=" * 80)
print(" ЧАСТЬ 1 ЗАВЕРШЕНА: Данные загружены, очищены и сбалансированы")
print("=" * 80)
print("\nСледующие шаги:")
print("  Часть 2: Обучение моделей (RandomForest, XGBoost)")
print("  Часть 3: Оценка моделей и визуализация (Confusion Matrix, ROC)")
print("  Часть 4: SHAP интерпретация и Radar Chart")

from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score

print("\n" + "=" * 80)
print("ЧАСТЬ 2: ОБУЧЕНИЕ МОДЕЛЕЙ С НАСТРОЙКОЙ ГИПЕРПАРАМЕТРОВ")
print("=" * 80)

# ======================================================================================
# 11. BASELINE МОДЕЛЬ (RandomForest с дефолтными параметрами)
# ======================================================================================
print("\n" + "=" * 80)
print("11. BASELINE МОДЕЛЬ (RandomForest с дефолтными параметрами)")
print("=" * 80)

rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Дополнительная защита от дисбаланса
)

print("Обучение Baseline модели...")
rf_baseline.fit(X_train, y_train)

# Оценка на Train
y_train_pred_baseline = rf_baseline.predict(X_train)
train_acc_baseline = accuracy_score(y_train, y_train_pred_baseline)
train_f1_baseline = f1_score(y_train, y_train_pred_baseline, average='weighted')

# Оценка на Test
y_test_pred_baseline = rf_baseline.predict(X_test)
test_acc_baseline = accuracy_score(y_test, y_test_pred_baseline)
test_f1_baseline = f1_score(y_test, y_test_pred_baseline, average='weighted')

print(f"\n{'Метрика':<25} {'Train':>12} {'Test':>12}")
print("-" * 50)
print(f"{'Accuracy':<25} {train_acc_baseline:>12.4f} {test_acc_baseline:>12.4f}")
print(f"{'F1-Score (weighted)':<25} {train_f1_baseline:>12.4f} {test_f1_baseline:>12.4f}")

# Детализация по классам
print("\nClassification Report (Test) - Baseline:")
print(classification_report(
    y_test,
    y_test_pred_baseline,
    target_names=[class_mapping[i] for i in sorted(class_mapping.keys())],
    digits=4
))

# ======================================================================================
# 12. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ RandomForest (GridSearchCV)
# ======================================================================================
print("\n" + "=" * 80)
print("12. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ RandomForest (GridSearchCV)")
print("=" * 80)

# Параметры для поиска
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

print(f"Сетка параметров:")
for param, values in param_grid_rf.items():
    print(f"  • {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid_rf.values()])
print(f"\n Всего комбинаций: {total_combinations}")
print(f" Кросс-валидация: StratifiedKFold (5 фолдов)")
print(f" Метрика оптимизации: F1-Score (weighted)")

# Кросс-валидация
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Скоринг
f1_scorer = make_scorer(f1_score, average='weighted')

# GridSearchCV
print("\nЗапуск GridSearchCV... (это может занять несколько минут)")
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid_rf,
    cv=cv_strategy,
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search_rf.fit(X_train, y_train)

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ GridSearchCV (RandomForest):")
print("=" * 80)

# Лучшие параметры
print("\nЛучшие параметры:")
for param, value in grid_search_rf.best_params_.items():
    print(f"  • {param}: {value}")

print(f"\n Лучший F1-Score (CV): {grid_search_rf.best_score_:.4f}")

# Топ-5 комбинаций
cv_results_rf = pd.DataFrame(grid_search_rf.cv_results_)
top5_rf = cv_results_rf.nsmallest(5, 'rank_test_score')[
    ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
]

print("\nТоп-5 лучших комбинаций:")
for idx, row in top5_rf.iterrows():
    print(f"\n  Ранг {int(row['rank_test_score'])}:")
    print(f"    F1-Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    print(f"    Параметры: {row['params']}")

# Сохраняем полные результаты
cv_results_rf.to_csv(
    f'{output_dir}/02_GridSearch_RF_results.csv',
    index=False,
    encoding='utf-8-sig'
)
print(f"\n Сохранены результаты: 02_GridSearch_RF_results.csv")

# ======================================================================================
# 13. ОЦЕНКА ОПТИМИЗИРОВАННОГО RandomForest НА ТЕСТЕ
# ======================================================================================
print("\n" + "=" * 80)
print("13. ОЦЕНКА ОПТИМИЗИРОВАННОГО RandomForest")
print("=" * 80)

# Лучшая модель
best_rf = grid_search_rf.best_estimator_

# Предсказания
y_train_pred_rf = best_rf.predict(X_train)
y_test_pred_rf = best_rf.predict(X_test)

# Метрики Train
train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
train_f1_rf = f1_score(y_train, y_train_pred_rf, average='weighted')

# Метрики Test
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)
test_f1_rf = f1_score(y_test, y_test_pred_rf, average='weighted')

print(f"\n{'Модель':<30} {'Train Acc':>12} {'Test Acc':>12} {'Train F1':>12} {'Test F1':>12}")
print("-" * 80)
print(f"{'Baseline RF':<30} {train_acc_baseline:>12.4f} {test_acc_baseline:>12.4f} "
      f"{train_f1_baseline:>12.4f} {test_f1_baseline:>12.4f}")
print(f"{'Optimized RF (GridSearch)':<30} {train_acc_rf:>12.4f} {test_acc_rf:>12.4f} "
      f"{train_f1_rf:>12.4f} {test_f1_rf:>12.4f}")

improvement_acc = (test_acc_rf - test_acc_baseline) / test_acc_baseline * 100
improvement_f1 = (test_f1_rf - test_f1_baseline) / test_f1_baseline * 100

print(f"\n Улучшение Accuracy: {improvement_acc:+.2f}%")
print(f" Улучшение F1-Score: {improvement_f1:+.2f}%")

# Classification Report
print("\n" + "=" * 80)
print("Classification Report (Test) - Optimized RandomForest:")
print("=" * 80)
report_rf = classification_report(
    y_test,
    y_test_pred_rf,
    target_names=[class_mapping[i] for i in sorted(class_mapping.keys())],
    digits=4,
    output_dict=True
)
print(classification_report(
    y_test,
    y_test_pred_rf,
    target_names=[class_mapping[i] for i in sorted(class_mapping.keys())],
    digits=4
))

# Сохраняем отчет
report_rf_df = pd.DataFrame(report_rf).transpose()
report_rf_df.to_csv(f'{output_dir}/03_Classification_Report_RF.csv', encoding='utf-8-sig')
print(f" Сохранен: 03_Classification_Report_RF.csv")

# ======================================================================================
# 14. ОБУЧЕНИЕ XGBoost (с настройкой гиперпараметров)
# ======================================================================================
print("\n" + "=" * 80)
print("14. ОБУЧЕНИЕ XGBoost С НАСТРОЙКОЙ ГИПЕРПАРАМЕТРОВ")
print("=" * 80)

# Параметры для XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

print(f"Сетка параметров XGBoost:")
for param, values in param_grid_xgb.items():
    print(f"  • {param}: {values}")

total_combinations_xgb = np.prod([len(v) for v in param_grid_xgb.values()])
print(f"\n Всего комбинаций: {total_combinations_xgb}")
print(f"   Это может занять МНОГО времени (>1 час).")
print(f"   Рекомендация: используйте RandomizedSearchCV вместо GridSearchCV")

# Используем RandomizedSearchCV для ускорения
from sklearn.model_selection import RandomizedSearchCV

print("\n Используем RandomizedSearchCV (50 случайных комбинаций)")

random_search_xgb = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',  # Для мультиклассовой классификации
        use_label_encoder=False,
        n_jobs=-1
    ),
    param_distributions=param_grid_xgb,
    n_iter=50,  # 50 случайных комбинаций
    cv=cv_strategy,
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

print("\nЗапуск RandomizedSearchCV для XGBoost...")
random_search_xgb.fit(X_train, y_train)

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ RandomizedSearchCV (XGBoost):")
print("=" * 80)

# Лучшие параметры
print("\nЛучшие параметры:")
for param, value in random_search_xgb.best_params_.items():
    print(f"  • {param}: {value}")

print(f"\n Лучший F1-Score (CV): {random_search_xgb.best_score_:.4f}")

# Топ-5 комбинаций
cv_results_xgb = pd.DataFrame(random_search_xgb.cv_results_)
top5_xgb = cv_results_xgb.nsmallest(5, 'rank_test_score')[
    ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
]

print("\nТоп-5 лучших комбинаций:")
for idx, row in top5_xgb.iterrows():
    print(f"\n  Ранг {int(row['rank_test_score'])}:")
    print(f"    F1-Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    print(f"    Параметры: {row['params']}")

# Сохраняем результаты
cv_results_xgb.to_csv(
    f'{output_dir}/04_RandomSearch_XGB_results.csv',
    index=False,
    encoding='utf-8-sig'
)
print(f"\n Сохранены результаты: 04_RandomSearch_XGB_results.csv")

# ======================================================================================
# 15. ОЦЕНКА XGBoost НА ТЕСТЕ
# ======================================================================================
print("\n" + "=" * 80)
print("15. ОЦЕНКА XGBoost НА ТЕСТЕ")
print("=" * 80)

# Лучшая модель XGBoost
best_xgb = random_search_xgb.best_estimator_

# Предсказания
y_train_pred_xgb = best_xgb.predict(X_train)
y_test_pred_xgb = best_xgb.predict(X_test)

# Метрики Train
train_acc_xgb = accuracy_score(y_train, y_train_pred_xgb)
train_f1_xgb = f1_score(y_train, y_train_pred_xgb, average='weighted')

# Метрики Test
test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)
test_f1_xgb = f1_score(y_test, y_test_pred_xgb, average='weighted')

print(f"\n{'Модель':<30} {'Train Acc':>12} {'Test Acc':>12} {'Train F1':>12} {'Test F1':>12}")
print("-" * 80)
print(f"{'Baseline RF':<30} {train_acc_baseline:>12.4f} {test_acc_baseline:>12.4f} "
      f"{train_f1_baseline:>12.4f} {test_f1_baseline:>12.4f}")
print(f"{'Optimized RF':<30} {train_acc_rf:>12.4f} {test_acc_rf:>12.4f} "
      f"{train_f1_rf:>12.4f} {test_f1_rf:>12.4f}")
print(f"{'Optimized XGBoost':<30} {train_acc_xgb:>12.4f} {test_acc_xgb:>12.4f} "
      f"{train_f1_xgb:>12.4f} {test_f1_xgb:>12.4f}")

# Classification Report
print("\n" + "=" * 80)
print("Classification Report (Test) - XGBoost:")
print("=" * 80)
report_xgb = classification_report(
    y_test,
    y_test_pred_xgb,
    target_names=[class_mapping[i] for i in sorted(class_mapping.keys())],
    digits=4,
    output_dict=True
)
print(classification_report(
    y_test,
    y_test_pred_xgb,
    target_names=[class_mapping[i] for i in sorted(class_mapping.keys())],
    digits=4
))

# Сохраняем отчет
report_xgb_df = pd.DataFrame(report_xgb).transpose()
report_xgb_df.to_csv(f'{output_dir}/05_Classification_Report_XGB.csv', encoding='utf-8-sig')
print(f" Сохранен: 05_Classification_Report_XGB.csv")

# ======================================================================================
# 16. ВЫБОР ФИНАЛЬНОЙ МОДЕЛИ
# ======================================================================================
print("\n" + "=" * 80)
print("16. ВЫБОР ФИНАЛЬНОЙ МОДЕЛИ")
print("=" * 80)

# Сравнение моделей
models_comparison = pd.DataFrame({
    'Модель': ['Baseline RF', 'Optimized RF', 'Optimized XGBoost'],
    'Test Accuracy': [test_acc_baseline, test_acc_rf, test_acc_xgb],
    'Test F1-Score': [test_f1_baseline, test_f1_rf, test_f1_xgb],
    'CV F1-Score': [
        train_f1_baseline,  # Приблизительно (нет CV для baseline)
        grid_search_rf.best_score_,
        random_search_xgb.best_score_
    ]
})

print("\nСводная таблица моделей:")
print(models_comparison.to_string(index=False))

# Выбираем лучшую модель по Test F1-Score
best_model_idx = models_comparison['Test F1-Score'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Модель']
best_model_f1 = models_comparison.loc[best_model_idx, 'Test F1-Score']

print(f"\n" + "=" * 80)
print(f" ФИНАЛЬНАЯ МОДЕЛЬ: {best_model_name}")
print(f"=" * 80)
print(f"Test F1-Score: {best_model_f1:.4f}")

if best_model_name == 'Optimized RF':
    final_model = best_rf
    final_predictions_test = y_test_pred_rf
    final_predictions_train = y_train_pred_rf
elif best_model_name == 'Optimized XGBoost':
    final_model = best_xgb
    final_predictions_test = y_test_pred_xgb
    final_predictions_train = y_train_pred_xgb
else:
    final_model = rf_baseline
    final_predictions_test = y_test_pred_baseline
    final_predictions_train = y_train_pred_baseline

# Сохраняем сравнение
models_comparison.to_csv(f'{output_dir}/06_Models_Comparison.csv', index=False, encoding='utf-8-sig')
print(f"\n Сохранено: 06_Models_Comparison.csv")

# Сохраняем финальную модель
import joblib
joblib.dump(final_model, f'{output_dir}/FINAL_MODEL.pkl')
print(f" Сохранена модель: FINAL_MODEL.pkl")

# Сохраняем scaler и список признаков
joblib.dump(scaler, f'{output_dir}/scaler.pkl')
with open(f'{output_dir}/feature_names.json', 'w', encoding='utf-8') as f:
    json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)
print(f" Сохранены: scaler.pkl, feature_names.json")

# ======================================================================================
# 17. КРОСС-ВАЛИДАЦИЯ ФИНАЛЬНОЙ МОДЕЛИ (детальная)
# ======================================================================================
print("\n" + "=" * 80)
print("17. ДЕТАЛЬНАЯ КРОСС-ВАЛИДАЦИЯ ФИНАЛЬНОЙ МОДЕЛИ")
print("=" * 80)

cv_scores_accuracy = cross_val_score(
    final_model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1
)
cv_scores_f1 = cross_val_score(
    final_model, X_train, y_train, cv=cv_strategy, scoring=f1_scorer, n_jobs=-1
)

print(f"\nКросс-валидация (5 фолдов):")
print(f"\nAccuracy по фолдам:")
for i, score in enumerate(cv_scores_accuracy, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Среднее: {cv_scores_accuracy.mean():.4f} (±{cv_scores_accuracy.std():.4f})")

print(f"\nF1-Score по фолдам:")
for i, score in enumerate(cv_scores_f1, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Среднее: {cv_scores_f1.mean():.4f} (±{cv_scores_f1.std():.4f})")

# Визуализация CV
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
axes[0].bar(range(1, 6), cv_scores_accuracy, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axhline(cv_scores_accuracy.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {cv_scores_accuracy.mean():.4f}')
axes[0].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title(f'Кросс-валидация: Accuracy ({best_model_name})', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(1, 6))
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# F1-Score
axes[1].bar(range(1, 6), cv_scores_f1, alpha=0.7, color='seagreen', edgecolor='black')
axes[1].axhline(cv_scores_f1.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {cv_scores_f1.mean():.4f}')
axes[1].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[1].set_ylabel('F1-Score (weighted)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Кросс-валидация: F1-Score ({best_model_name})', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, 6))
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/07_Cross_Validation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n Сохранен график: 07_Cross_Validation.png")

# Сохраняем CV результаты
cv_results_final = pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': cv_scores_accuracy,
    'F1-Score': cv_scores_f1
})
cv_results_final.loc['Mean'] = ['Mean', cv_scores_accuracy.mean(), cv_scores_f1.mean()]
cv_results_final.loc['Std'] = ['Std', cv_scores_accuracy.std(), cv_scores_f1.std()]
cv_results_final.to_csv(f'{output_dir}/08_CV_Results_Final.csv', encoding='utf-8-sig')
print(f" Сохранено: 08_CV_Results_Final.csv")

print("\n" + "=" * 80)
print(" ЧАСТЬ 2 ЗАВЕРШЕНА: Модели обучены и оптимизированы")
print("=" * 80)
print(f"\nФинальная модель: {best_model_name}")
print(f"Test F1-Score: {best_model_f1:.4f}")
print(f"Test Accuracy: {models_comparison.loc[best_model_idx, 'Test Accuracy']:.4f}")
print("\nСледующие шаги:")
print("  Часть 3: Визуализация (Confusion Matrix, Feature Importance)")
print("  Часть 4: SHAP интерпретация и Radar Chart")

# ======================================================================================
# ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (Confusion Matrix, ROC, Feature Importance)
# ======================================================================================

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

print("\n" + "=" * 80)
print("ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ МОДЕЛИ")
print("=" * 80)

# ======================================================================================
# 18. CONFUSION MATRIX (Тепловая карта)
# ======================================================================================
print("\n" + "=" * 80)
print("18. CONFUSION MATRIX")
print("=" * 80)

# Вычисляем Confusion Matrix для Test
cm = confusion_matrix(y_test, final_predictions_test)

# Нормализованная версия (проценты)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print("\nConfusion Matrix (абсолютные значения):")
print(cm)

print("\nConfusion Matrix (проценты по строкам):")
print(cm_normalized.round(2))

# Визуализация: 2 тепловые карты (абсолютные + проценты)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

class_names = [class_mapping[i] for i in sorted(class_mapping.keys())]

# 1. Абсолютные значения
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    square=True,
    linewidths=2,
    linecolor='black',
    cbar_kws={'label': 'Количество пациентов'},
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[0],
    annot_kws={'fontsize': 14, 'fontweight': 'bold'}
)
axes[0].set_xlabel('Предсказанный класс', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Истинный класс', fontsize=13, fontweight='bold')
axes[0].set_title(f'Confusion Matrix (Абсолютные значения)\n{best_model_name}',
                  fontsize=14, fontweight='bold')

# 2. Проценты
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.1f',
    cmap='Greens',
    square=True,
    linewidths=2,
    linecolor='black',
    cbar_kws={'label': 'Процент (%)'},
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1],
    annot_kws={'fontsize': 14, 'fontweight': 'bold'}
)
axes[1].set_xlabel('Предсказанный класс', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Истинный класс', fontsize=13, fontweight='bold')
axes[1].set_title(f'Confusion Matrix (Проценты по строкам)\n{best_model_name}',
                  fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/09_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n Сохранен: 09_Confusion_Matrix.png")

# Анализ ошибок
print("\n" + "=" * 80)
print("АНАЛИЗ ОШИБОК КЛАССИФИКАЦИИ:")
print("=" * 80)

for i, true_class in enumerate(class_names):
    total_true = cm[i].sum()
    correct = cm[i, i]
    accuracy_class = correct / total_true * 100

    print(f"\n{true_class}:")
    print(f"  Всего: {total_true}")
    print(f"  Правильно классифицировано: {correct} ({accuracy_class:.1f}%)")

    # Ошибки
    errors = []
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            errors.append(f"{cm[i, j]} → {class_names[j]}")

    if errors:
        print(f"  Ошибки: {', '.join(errors)}")
    else:
        print(f"   Без ошибок!")

# Сохраняем матрицу
cm_df = pd.DataFrame(
    cm,
    index=[f'True: {name}' for name in class_names],
    columns=[f'Pred: {name}' for name in class_names]
)
cm_df.to_csv(f'{output_dir}/10_Confusion_Matrix.csv', encoding='utf-8-sig')
print(f"\n Сохранено: 10_Confusion_Matrix.csv")

# ======================================================================================
# 19. ROC-AUC КРИВЫЕ (One-vs-Rest для мультикласса)
# ======================================================================================
print("\n" + "=" * 80)
print("19. ROC-AUC КРИВЫЕ (One-vs-Rest)")
print("=" * 80)

# Получаем вероятности предсказаний
if hasattr(final_model, 'predict_proba'):
    y_proba_test = final_model.predict_proba(X_test)

    # Бинаризация целевой переменной для One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    # Вычисляем ROC-AUC для каждого класса
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print("\nROC-AUC по классам:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:20}: {roc_auc[i]:.4f}")
    print(f"\n  {'Micro-average':20}: {roc_auc['micro']:.4f}")
    print(f"  {'Macro-average':20}: {roc_auc['macro']:.4f}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1. ROC кривые для каждого класса
    colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Красный, Оранжевый, Зеленый

    for i, color, class_name in zip(range(n_classes), colors, class_names):
        axes[0].plot(
            fpr[i], tpr[i], color=color, lw=3,
            label=f'{class_name} (AUC = {roc_auc[i]:.3f})'
        )

    axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Случайный классификатор')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    axes[0].set_title(f'ROC кривые (One-vs-Rest)\n{best_model_name}',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 2. Micro и Macro average
    axes[1].plot(
        fpr["micro"], tpr["micro"],
        color='deeppink', lw=3, linestyle='-',
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})'
    )
    axes[1].plot(
        fpr["macro"], tpr["macro"],
        color='navy', lw=3, linestyle='-',
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})'
    )
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Случайный классификатор')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    axes[1].set_title(f'ROC: Усредненные метрики\n{best_model_name}',
                      fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_ROC_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n Сохранен: 11_ROC_Curves.png")

    # Сохраняем AUC значения
    auc_df = pd.DataFrame({
        'Класс': class_names + ['Micro-average', 'Macro-average'],
        'AUC': [roc_auc[i] for i in range(n_classes)] + [roc_auc['micro'], roc_auc['macro']]
    })
    auc_df.to_csv(f'{output_dir}/12_ROC_AUC_scores.csv', index=False, encoding='utf-8-sig')
    print(f" Сохранено: 12_ROC_AUC_scores.csv")

else:
    print(" Модель не поддерживает predict_proba. ROC-AUC пропущен.")

# ======================================================================================
# 20. FEATURE IMPORTANCE (Важность признаков)
# ======================================================================================
print("\n" + "=" * 80)
print("20. FEATURE IMPORTANCE (Важность признаков)")
print("=" * 80)

if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feature_names = X_train.columns

    # Сортируем по важности
    indices = np.argsort(importances)[::-1]

    # Топ-30 признаков
    top_n = min(30, len(importances))
    top_indices = indices[:top_n]

    print(f"\nТоп-{top_n} самых важных признаков:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2}. {feature_names[idx]:50} : {importances[idx]:.6f}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 1. Топ-30 горизонтальный bar chart
    y_pos = np.arange(top_n)
    colors_importance = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

    axes[0].barh(
        y_pos,
        importances[top_indices],
        color=colors_importance,
        edgecolor='black',
        linewidth=0.8
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([feature_names[i][:40] for i in top_indices], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Важность', fontsize=13, fontweight='bold')
    axes[0].set_title(f'Топ-{top_n} важных признаков\n{best_model_name}',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # 2. Кумулятивная важность
    cumulative_importances = np.cumsum(importances[indices])

    axes[1].plot(
        range(1, len(cumulative_importances) + 1),
        cumulative_importances * 100,
        'b-',
        linewidth=2.5
    )
    axes[1].axhline(y=80, color='r', linestyle='--', linewidth=2, label='80% порог')
    axes[1].axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% порог')

    # Находим количество признаков для 80% и 90%
    n_for_80 = np.argmax(cumulative_importances >= 0.80) + 1
    n_for_90 = np.argmax(cumulative_importances >= 0.90) + 1

    axes[1].axvline(x=n_for_80, color='r', linestyle=':', linewidth=2, alpha=0.5)
    axes[1].axvline(x=n_for_90, color='orange', linestyle=':', linewidth=2, alpha=0.5)

    axes[1].text(n_for_80, 5, f'{n_for_80} признаков', rotation=90,
                 fontsize=10, color='red', fontweight='bold')
    axes[1].text(n_for_90, 5, f'{n_for_90} признаков', rotation=90,
                 fontsize=10, color='orange', fontweight='bold')

    axes[1].set_xlabel('Количество признаков', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Кумулятивная важность (%)', fontsize=13, fontweight='bold')
    axes[1].set_title(f'Кумулятивная важность признаков\n{best_model_name}',
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_Feature_Importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n Сохранен: 13_Feature_Importance.png")

    print(f"\n Статистика важности:")
    print(f"  • {n_for_80} признаков объясняют 80% важности")
    print(f"  • {n_for_90} признаков объясняют 90% важности")
    print(f"  • Всего признаков: {len(importances)}")

    # Сохраняем таблицу важности
    importance_df = pd.DataFrame({
        'Признак': feature_names,
        'Важность': importances,
        'Ранг': np.argsort(np.argsort(importances)[::-1]) + 1
    }).sort_values('Важность', ascending=False)

    importance_df.to_csv(f'{output_dir}/14_Feature_Importance_Full.csv',
                         index=False, encoding='utf-8-sig')
    print(f" Сохранено: 14_Feature_Importance_Full.csv")

else:
    print(" Модель не имеет атрибута feature_importances_. Пропущено.")

# ======================================================================================
# 21. РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ ПО КЛАССАМ
# ======================================================================================
print("\n" + "=" * 80)
print("21. АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПРЕДСКАЗАНИЙ")
print("=" * 80)

# Распределение предсказаний
pred_distribution = pd.Series(final_predictions_test).value_counts().sort_index()
true_distribution = y_test.value_counts().sort_index()

print("\nСравнение истинного и предсказанного распределения:")
print(f"{'Класс':<25} {'Истинное':>12} {'Предсказано':>12} {'Разница':>12}")
print("-" * 65)

for cls in sorted(class_mapping.keys()):
    true_count = true_distribution.get(cls, 0)
    pred_count = pred_distribution.get(cls, 0)
    diff = pred_count - true_count

    print(f"{class_mapping[cls]:<25} {true_count:>12} {pred_count:>12} {diff:>+12}")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_pos = np.arange(len(class_names))
width = 0.35

# 1. Сравнение распределений
axes[0].bar(
    x_pos - width / 2,
    [true_distribution.get(i, 0) for i in range(len(class_names))],
    width,
    label='Истинное распределение',
    color='steelblue',
    alpha=0.8,
    edgecolor='black',
    linewidth=1.5
)
axes[0].bar(
    x_pos + width / 2,
    [pred_distribution.get(i, 0) for i in range(len(class_names))],
    width,
    label='Предсказанное распределение',
    color='coral',
    alpha=0.8,
    edgecolor='black',
    linewidth=1.5
)

axes[0].set_xlabel('Класс', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Количество примеров', fontsize=13, fontweight='bold')
axes[0].set_title('Сравнение истинного и предсказанного распределения',
                  fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(class_names, fontsize=11)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# 2. Матрица ошибок в виде пропорций
cm_prop = cm.astype('float') / cm.sum() * 100  # Процент от всех предсказаний

im = axes[1].imshow(cm_prop, cmap='YlOrRd', aspect='auto')

# Аннотации
for i in range(len(class_names)):
    for j in range(len(class_names)):
        text_color = 'white' if cm_prop[i, j] > cm_prop.max() / 2 else 'black'
        axes[1].text(j, i, f'{cm_prop[i, j]:.1f}%\n({cm[i, j]})',
                     ha="center", va="center",
                     color=text_color, fontsize=12, fontweight='bold')

axes[1].set_xticks(np.arange(len(class_names)))
axes[1].set_yticks(np.arange(len(class_names)))
axes[1].set_xticklabels(class_names, fontsize=11)
axes[1].set_yticklabels(class_names, fontsize=11)
axes[1].set_xlabel('Предсказанный класс', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Истинный класс', fontsize=13, fontweight='bold')
axes[1].set_title('Confusion Matrix (% от всех предсказаний)',
                  fontsize=14, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=axes[1])
cbar.set_label('Процент (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/15_Predictions_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n Сохранен: 15_Predictions_Distribution.png")

# ======================================================================================
# 22. ТЕКСТОВЫЙ ОТЧЕТ (REPORT.txt)
# ======================================================================================
print("\n" + "=" * 80)
print("22. ГЕНЕРАЦИЯ ТЕКСТОВОГО ОТЧЕТА")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("ОТЧЕТ ПО ФАКТОРНОМУ АНАЛИЗУ ВОССТАНОВЛЕНИЯ ПОСЛЕ ИНФАРКТА МИОКАРДА")
report_lines.append("=" * 80)
report_lines.append(f"\nДата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Папка результатов: {output_dir}")
report_lines.append("\n" + "=" * 80)
report_lines.append("1. ДАННЫЕ")
report_lines.append("=" * 80)
report_lines.append(f"\nИсходный датасет: {file_path}")
report_lines.append(f"Всего пациентов (после очистки): {len(X_scaled)}")
report_lines.append(f"Количество признаков: {X_train.shape[1]}")
report_lines.append(f"  - Числовых: {len(numerical_features)}")
report_lines.append(f"  - Категориальных (до One-Hot): {len(categorical_features)}")

report_lines.append("\n" + "=" * 80)
report_lines.append("2. КЛАССЫ")
report_lines.append("=" * 80)
report_lines.append("\nРаспределение классов (после SMOTE):")
for cls, count in y_resampled.value_counts().sort_index().items():
    report_lines.append(f"  Class {int(cls)} ({class_mapping[int(cls)]:<20}): {count}")

report_lines.append("\n" + "=" * 80)
report_lines.append("3. ОБУЧЕНИЕ МОДЕЛЕЙ")
report_lines.append("=" * 80)
report_lines.append(f"\nФинальная модель: {best_model_name}")
report_lines.append(f"\nСравнение моделей:")
report_lines.append(models_comparison.to_string(index=False))

report_lines.append("\n" + "=" * 80)
report_lines.append("4. МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ")
report_lines.append("=" * 80)
report_lines.append(f"\nTest Accuracy: {models_comparison.loc[best_model_idx, 'Test Accuracy']:.4f}")
report_lines.append(f"Test F1-Score: {best_model_f1:.4f}")

report_lines.append("\n\nClassification Report (Test):")
report_lines.append(classification_report(
    y_test,
    final_predictions_test,
    target_names=class_names,
    digits=4
))

report_lines.append("\n" + "=" * 80)
report_lines.append("5. CONFUSION MATRIX")
report_lines.append("=" * 80)
report_lines.append("\nАбсолютные значения:")
report_lines.append(str(cm_df))

report_lines.append("\n" + "=" * 80)
report_lines.append("6. ТОП-20 ВАЖНЫХ ПРИЗНАКОВ")
report_lines.append("=" * 80)
if hasattr(final_model, 'feature_importances_'):
    top_20_features = importance_df.head(20)
    report_lines.append("\n" + top_20_features.to_string(index=False))

report_lines.append("\n\n" + "=" * 80)
report_lines.append("7. ROC-AUC SCORES")
report_lines.append("=" * 80)
if hasattr(final_model, 'predict_proba'):
    for i, class_name in enumerate(class_names):
        report_lines.append(f"\n  {class_name:20}: {roc_auc[i]:.4f}")
    report_lines.append(f"\n  {'Micro-average':20}: {roc_auc['micro']:.4f}")
    report_lines.append(f"  {'Macro-average':20}: {roc_auc['macro']:.4f}")

report_lines.append("\n\n" + "=" * 80)
report_lines.append("КОНЕЦ ОТЧЕТА")
report_lines.append("=" * 80)

# Сохраняем отчет
report_text = '\n'.join(report_lines)
with open(f'{output_dir}/REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n Сохранен: REPORT.txt")
print(f"\n Отчет содержит:")
print(f"  • Описание данных")
print(f"  • Сравнение моделей")
print(f"  • Метрики финальной модели")
print(f"  • Confusion Matrix")
print(f"  • Топ-20 важных признаков")
print(f"  • ROC-AUC scores")

print("\n" + "=" * 80)
print(" ЧАСТЬ 3 ЗАВЕРШЕНА: Визуализация результатов готова")
print("=" * 80)
print(f"\nСохранено в {output_dir}:")
print("  • 09_Confusion_Matrix.png")
print("  • 11_ROC_Curves.png")
print("  • 13_Feature_Importance.png")
print("  • 15_Predictions_Distribution.png")
print("  • REPORT.txt")
print("\nСледующие шаги:")
print("  Часть 4: SHAP интерпретация и Radar Chart для профилей классов")

# ======================================================================================
# ЧАСТЬ 4: SHAP ИНТЕРПРЕТАЦИЯ И RADAR CHART ДЛЯ ПРОФИЛЕЙ КЛАССОВ
# ======================================================================================

print("\n" + "=" * 80)
print("ЧАСТЬ 4: SHAP ИНТЕРПРЕТАЦИЯ И RADAR CHART")
print("=" * 80)

# ======================================================================================
# 23. SHAP VALUES (Интерпретация модели)
# ======================================================================================
print("\n" + "=" * 80)
print("23. SHAP VALUES - ИНТЕРПРЕТАЦИЯ МОДЕЛИ")
print("=" * 80)

try:
    import shap

    shap_available = True
    print(" Библиотека SHAP доступна")
except ImportError:
    shap_available = False
    print(" Библиотека SHAP не установлена.")
    print("   Установите: pip install shap")

if shap_available:
    print("\nИнициализация SHAP Explainer...")
    print("(это может занять несколько минут для большого количества признаков)")

    # Выбираем подвыборку для SHAP (для ускорения)
    # SHAP очень медленный на больших данных
    sample_size = min(200, len(X_test))
    X_test_sample = X_test.iloc[:sample_size]
    y_test_sample = y_test.iloc[:sample_size]

    print(f" Используем выборку из {sample_size} примеров для SHAP")

    # TreeExplainer для tree-based моделей (быстрее)
    if 'RandomForest' in best_model_name or 'XGBoost' in best_model_name:
        explainer = shap.TreeExplainer(final_model)
        print(" Используем TreeExplainer (оптимизирован для деревьев)")
    else:
        explainer = shap.Explainer(final_model, X_train)
        print(" Используем базовый Explainer")

    print("\nВычисление SHAP values...")
    shap_values = explainer.shap_values(X_test_sample)

    print(" SHAP values вычислены!")
    print(f"  Форма: {np.array(shap_values).shape}")

    # ======================================================================================
    # 24. SHAP SUMMARY PLOT (Beeswarm)
    # ======================================================================================
    print("\n" + "=" * 80)
    print("24. SHAP SUMMARY PLOT (Beeswarm)")
    print("=" * 80)

    # Для мультикласса: визуализируем каждый класс отдельно
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for i, class_name in enumerate(class_names):
        plt.sca(axes[i])

        # SHAP values для класса i
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[i]
        else:
            shap_vals_class = shap_values[:, :, i]

        shap.summary_plot(
            shap_vals_class,
            X_test_sample,
            plot_type="dot",
            show=False,
            max_display=15,
            plot_size=None
        )
        axes[i].set_title(f'SHAP: {class_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/16_SHAP_Beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Сохранен: 16_SHAP_Beeswarm.png")

    # ======================================================================================
    # 25. SHAP BAR PLOT (Средняя важность)
    # ======================================================================================
    print("\n" + "=" * 80)
    print("25. SHAP BAR PLOT (Средняя важность по классам)")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for i, class_name in enumerate(class_names):
        plt.sca(axes[i])

        if isinstance(shap_values, list):
            shap_vals_class = shap_values[i]
        else:
            shap_vals_class = shap_values[:, :, i]

        shap.summary_plot(
            shap_vals_class,
            X_test_sample,
            plot_type="bar",
            show=False,
            max_display=20,
            plot_size=None
        )
        axes[i].set_title(f'SHAP Важность: {class_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/17_SHAP_Bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Сохранен: 17_SHAP_Bar.png")

    # ======================================================================================
    # 26. SHAP WATERFALL PLOTS (Примеры из каждого класса)
    # ======================================================================================
    print("\n" + "=" * 80)
    print("26. SHAP WATERFALL PLOTS (Примеры пациентов)")
    print("=" * 80)

    # Находим по одному типичному примеру из каждого класса
    waterfall_examples = []
    for cls in sorted(class_mapping.keys()):
        # Ищем правильно классифицированный пример
        mask = (y_test_sample == cls)
        if mask.sum() > 0:
            idx = mask[mask].index[0]  # Первый пример из этого класса
            sample_idx = X_test_sample.index.get_loc(idx)
            waterfall_examples.append((cls, sample_idx, idx))

    if len(waterfall_examples) > 0:
        fig, axes = plt.subplots(len(waterfall_examples), 1,
                                 figsize=(12, 6 * len(waterfall_examples)))

        if len(waterfall_examples) == 1:
            axes = [axes]

        for ax_idx, (cls, sample_idx, original_idx) in enumerate(waterfall_examples):
            plt.sca(axes[ax_idx])

            # Для TreeExplainer
            if isinstance(shap_values, list):
                shap_explanation = shap.Explanation(
                    values=shap_values[cls][sample_idx],
                    base_values=explainer.expected_value[cls],
                    data=X_test_sample.iloc[sample_idx].values,
                    feature_names=X_test_sample.columns.tolist()
                )
            else:
                shap_explanation = shap.Explanation(
                    values=shap_values[sample_idx, :, cls],
                    base_values=explainer.expected_value[cls],
                    data=X_test_sample.iloc[sample_idx].values,
                    feature_names=X_test_sample.columns.tolist()
                )

            shap.waterfall_plot(shap_explanation, max_display=15, show=False)
            axes[ax_idx].set_title(
                f'SHAP Waterfall: Пациент #{original_idx} - {class_mapping[cls]}',
                fontsize=14, fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig(f'{output_dir}/18_SHAP_Waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(" Сохранен: 18_SHAP_Waterfall.png")

    # ======================================================================================
    # 27. SHAP DEPENDENCE PLOTS (Топ-5 признаков)
    # ======================================================================================
    print("\n" + "=" * 80)
    print("27. SHAP DEPENDENCE PLOTS (Влияние топ-признаков)")
    print("=" * 80)

    if hasattr(final_model, 'feature_importances_'):
        # Берем топ-5 самых важных признаков
        top_5_indices = indices[:5]
        top_5_features = [feature_names[i] for i in top_5_indices]

        print(f"\nТоп-5 признаков для Dependence Plot:")
        for i, feat in enumerate(top_5_features, 1):
            print(f"  {i}. {feat}")

        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()

        for i, feat in enumerate(top_5_features):
            if i < 6:
                plt.sca(axes[i])

                # Dependence plot для класса "Смерть" (самый критичный)
                if isinstance(shap_values, list):
                    shap_vals_death = shap_values[0]  # Класс 0 = Смерть
                else:
                    shap_vals_death = shap_values[:, :, 0]

                feat_idx = X_test_sample.columns.get_loc(feat)

                shap.dependence_plot(
                    feat_idx,
                    shap_vals_death,
                    X_test_sample,
                    show=False,
                    ax=axes[i]
                )
                axes[i].set_title(f'{feat[:50]}', fontsize=12, fontweight='bold')

        # Убираем лишний subplot
        if len(top_5_features) < 6:
            fig.delaxes(axes[5])

        plt.suptitle('SHAP Dependence Plots (для класса "Смерть")',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/19_SHAP_Dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(" Сохранен: 19_SHAP_Dependence.png")

    # Сохраняем SHAP values для дальнейшего анализа
    if isinstance(shap_values, list):
        shap_df = pd.DataFrame(
            shap_values[0],  # Класс "Смерть"
            columns=X_test_sample.columns
        )
    else:
        shap_df = pd.DataFrame(
            shap_values[:, :, 0],  # Класс "Смерть"
            columns=X_test_sample.columns
        )

    shap_df.to_csv(f'{output_dir}/20_SHAP_values_Death_class.csv',
                   index=False, encoding='utf-8-sig')
    print(" Сохранено: 20_SHAP_values_Death_class.csv")

else:
    print("\n SHAP анализ пропущен. Установите библиотеку: pip install shap")

# ======================================================================================
# 28. RADAR CHART - ПРОФИЛИ ПАЦИЕНТОВ ПО КЛАССАМ
# ======================================================================================
print("\n" + "=" * 80)
print("28. RADAR CHART - СРАВНЕНИЕ ПРОФИЛЕЙ КЛАССОВ")
print("=" * 80)

# Выбираем ключевые числовые признаки для Radar Chart
# (категориальные плохо подходят для радара)
if len(numerical_features) > 0:
    # Берем топ-8 самых важных ЧИСЛОВЫХ признаков
    if hasattr(final_model, 'feature_importances_'):
        # Фильтруем только числовые признаки из важных
        important_numerical = []
        for idx in indices:
            feat_name = feature_names[idx]
            if feat_name in numerical_features:
                important_numerical.append(feat_name)
            if len(important_numerical) >= 8:
                break

        radar_features = important_numerical
    else:
        # Если нет feature_importances, берем первые 8 числовых
        radar_features = numerical_features[:8]

    print(f"\nПризнаки для Radar Chart ({len(radar_features)}):")
    for i, feat in enumerate(radar_features, 1):
        print(f"  {i}. {feat}")

    # Вычисляем средние значения по каждому классу
    # ВАЖНО: используем исходные (не стандартизированные) данные для интерпретации
    df_with_labels = X_encoded.copy()
    df_with_labels['Target'] = y

    class_profiles = {}
    for cls in sorted(class_mapping.keys()):
        class_data = df_with_labels[df_with_labels['Target'] == cls]
        profile = {}
        for feat in radar_features:
            if feat in class_data.columns:
                profile[feat] = class_data[feat].mean()
            else:
                profile[feat] = 0
        class_profiles[cls] = profile

    print("\nСредние значения признаков по классам:")
    for cls, profile in class_profiles.items():
        print(f"\n{class_mapping[cls]}:")
        for feat, val in profile.items():
            print(f"  {feat[:40]:40}: {val:8.2f}")

    # ======================================================================================
    # ВИЗУАЛИЗАЦИЯ RADAR CHART
    # ======================================================================================
    print("\nСоздание Radar Chart...")

    # Нормализация данных для радара (Min-Max в диапазон [0, 1])
    from sklearn.preprocessing import MinMaxScaler

    radar_data = []
    for cls in sorted(class_mapping.keys()):
        radar_data.append([class_profiles[cls][feat] for feat in radar_features])

    radar_data = np.array(radar_data)

    # Нормализуем по столбцам (каждый признак отдельно)
    scaler_radar = MinMaxScaler()
    radar_data_normalized = scaler_radar.fit_transform(radar_data.T).T

    # Создаем Radar Chart
    from math import pi

    # Количество переменных
    num_vars = len(radar_features)

    # Углы для каждой оси
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Замыкаем круг

    # Инициализация графика
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    # Цвета для классов
    colors_radar = ['#e74c3c', '#f39c12', '#2ecc71']  # Красный, Оранжевый, Зеленый

    # Рисуем профиль для каждого класса
    for idx, cls in enumerate(sorted(class_mapping.keys())):
        values = radar_data_normalized[idx].tolist()
        values += values[:1]  # Замыкаем круг

        ax.plot(angles, values, 'o-', linewidth=3,
                label=class_mapping[cls], color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    # Настройка осей
    ax.set_xticks(angles[:-1])

    # Сокращаем названия признаков для читаемости
    labels = [feat[:30] + '...' if len(feat) > 30 else feat for feat in radar_features]
    ax.set_xticklabels(labels, size=10, fontweight='bold')

    # Настройка радиальной оси
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.set_rlabel_position(0)

    # Легенда и заголовок
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Radar Chart: Профили пациентов по классам\n' +
              '(Нормализованные средние значения ключевых признаков)',
              size=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/21_Radar_Chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Сохранен: 21_Radar_Chart.png")

    # Сохраняем данные профилей
    profiles_df = pd.DataFrame(radar_data,
                               columns=radar_features,
                               index=[class_mapping[i] for i in sorted(class_mapping.keys())])
    profiles_df.to_csv(f'{output_dir}/22_Class_Profiles_Raw.csv', encoding='utf-8-sig')

    profiles_normalized_df = pd.DataFrame(
        radar_data_normalized,
        columns=radar_features,
        index=[class_mapping[i] for i in sorted(class_mapping.keys())]
    )
    profiles_normalized_df.to_csv(f'{output_dir}/23_Class_Profiles_Normalized.csv',
                                  encoding='utf-8-sig')

    print(" Сохранены профили классов:")
    print("  • 22_Class_Profiles_Raw.csv (исходные значения)")
    print("  • 23_Class_Profiles_Normalized.csv (нормализованные)")

else:
    print("\n Недостаточно числовых признаков для Radar Chart")

# ======================================================================================
# 29. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: VIOLIN PLOTS ПО КЛАССАМ
# ======================================================================================
print("\n" + "=" * 80)
print("29. VIOLIN PLOTS - РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ ПО КЛАССАМ")
print("=" * 80)

if len(numerical_features) > 0 and len(radar_features) > 0:
    # Берем топ-6 признаков для Violin plots
    violin_features = radar_features[:6]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # Подготовка данных
    df_violin = df_with_labels.copy()
    df_violin['Класс'] = df_violin['Target'].map(class_mapping)

    for i, feat in enumerate(violin_features):
        if feat in df_violin.columns:
            sns.violinplot(
                data=df_violin,
                x='Класс',
                y=feat,
                palette=['#e74c3c', '#f39c12', '#2ecc71'],
                ax=axes[i],
                inner='box'
            )

            axes[i].set_title(f'{feat[:40]}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Класс', fontsize=11, fontweight='bold')
            axes[i].set_ylabel('Значение', fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Распределение ключевых признаков по классам (Violin Plot)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/24_Violin_Plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Сохранен: 24_Violin_Plots.png")

# ======================================================================================
# 30. СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ КЛАССОВ (t-test / ANOVA)
# ======================================================================================
print("\n" + "=" * 80)
print("30. СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ КЛАССОВ")
print("=" * 80)

if len(numerical_features) > 0 and len(radar_features) > 0:
    from scipy import stats

    print("\nОднофакторный ANOVA для ключевых признаков:")
    print(f"{'Признак':<45} {'F-статистика':>15} {'p-value':>12} {'Значимость':>12}")
    print("-" * 85)

    anova_results = []

    for feat in radar_features:
        if feat in df_with_labels.columns:
            # Разбиваем данные по классам
            groups = [
                df_with_labels[df_with_labels['Target'] == cls][feat].dropna()
                for cls in sorted(class_mapping.keys())
            ]

            # Проверяем, есть ли данные в каждой группе
            if all(len(g) > 0 for g in groups):
                f_stat, p_value = stats.f_oneway(*groups)

                # Определяем значимость
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = 'n.s.'

                print(f"{feat[:45]:<45} {f_stat:>15.4f} {p_value:>12.6f} {significance:>12}")

                anova_results.append({
                    'Признак': feat,
                    'F-статистика': f_stat,
                    'p-value': p_value,
                    'Значимость': significance
                })

    print("\nЛегенда: *** p<0.001, ** p<0.01, * p<0.05, n.s. = незначимо")

    # Сохраняем результаты ANOVA
    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(f'{output_dir}/25_ANOVA_Results.csv',
                    index=False, encoding='utf-8-sig')
    print(f"\n Сохранено: 25_ANOVA_Results.csv")

# ======================================================================================
# 31. ФИНАЛЬНАЯ СВОДКА И КЛИНИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
# ======================================================================================
print("\n" + "=" * 80)
print("31. ФИНАЛЬНАЯ СВОДКА И КЛИНИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ")
print("=" * 80)

interpretation_lines = []
interpretation_lines.append("\n" + "=" * 80)
interpretation_lines.append("КЛИНИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
interpretation_lines.append("=" * 80)

interpretation_lines.append("\n1. ПРЕДСКАЗАТЕЛЬНАЯ СПОСОБНОСТЬ МОДЕЛИ:")
interpretation_lines.append(f"   - Модель: {best_model_name}")
interpretation_lines.append(f"   - Test Accuracy: {models_comparison.loc[best_model_idx, 'Test Accuracy']:.2%}")
interpretation_lines.append(f"   - Test F1-Score: {best_model_f1:.4f}")

interpretation_lines.append("\n2. КЛЮЧЕВЫЕ ФАКТОРЫ ВОССТАНОВЛЕНИЯ:")
if hasattr(final_model, 'feature_importances_'):
    interpretation_lines.append("   Топ-10 самых важных признаков:")
    for i in range(min(10, len(importance_df))):
        row = importance_df.iloc[i]
        interpretation_lines.append(f"   {i + 1:2}. {row['Признак'][:50]:<50} (важность: {row['Важность']:.4f})")

interpretation_lines.append("\n3. ХАРАКТЕРИСТИКИ КЛАССОВ:")
interpretation_lines.append("\n   А) КЛАСС 'СМЕРТЬ' (Class 0):")
interpretation_lines.append("      - Самый малочисленный класс (требовалась SMOTE)")
interpretation_lines.append("      - Критические факторы риска (по SHAP/Feature Importance):")
if hasattr(final_model, 'feature_importances_') and len(radar_features) > 0:
    death_profile = class_profiles[0]
    interpretation_lines.append(f"        • Средние значения ключевых показателей:")
    for feat in radar_features[:5]:
        interpretation_lines.append(f"          - {feat[:40]}: {death_profile[feat]:.2f}")

interpretation_lines.append("\n   Б) КЛАСС 'НЕ ВОССТАНОВИЛСЯ' (Class 1):")
interpretation_lines.append("      - Промежуточный класс")
if hasattr(final_model, 'feature_importances_') and len(radar_features) > 0:
    not_recovered_profile = class_profiles[1]
    interpretation_lines.append(f"        • Средние значения:")
    for feat in radar_features[:5]:
        interpretation_lines.append(f"          - {feat[:40]}: {not_recovered_profile[feat]:.2f}")

interpretation_lines.append("\n   В) КЛАСС 'ВОССТАНОВИЛСЯ' (Class 2):")
interpretation_lines.append("      - Положительный исход")
if hasattr(final_model, 'feature_importances_') and len(radar_features) > 0:
    recovered_profile = class_profiles[2]
    interpretation_lines.append(f"        • Средние значения:")
    for feat in radar_features[:5]:
        interpretation_lines.append(f"          - {feat[:40]}: {recovered_profile[feat]:.2f}")

interpretation_lines.append("\n4. ТОЧНОСТЬ ПРЕДСКАЗАНИЙ ПО КЛАССАМ:")
for i, class_name in enumerate(class_names):
    total_true = cm[i].sum()
    correct = cm[i, i]
    accuracy_class = correct / total_true * 100
    interpretation_lines.append(f"   {class_name:20}: {accuracy_class:.1f}% ({correct}/{total_true})")

interpretation_lines.append("\n5. РЕКОМЕНДАЦИИ ДЛЯ КЛИНИЧЕСКОЙ ПРАКТИКИ:")
interpretation_lines.append("   - Использовать топ-10 признаков для ранней стратификации риска")
interpretation_lines.append("   - Обратить внимание на пациентов с профилем класса 'Смерть'")
interpretation_lines.append("   - Модель может помочь в персонализации программ реабилитации")

if shap_available:
    interpretation_lines.append("\n6. ИНТЕРПРЕТИРУЕМОСТЬ (SHAP):")
    interpretation_lines.append("   - SHAP values показывают вклад каждого признака в предсказание")
    interpretation_lines.append("   - См. графики: 16_SHAP_Beeswarm.png, 19_SHAP_Dependence.png")
    interpretation_lines.append("   - Позволяет объяснить решение модели для каждого пациента")

interpretation_lines.append("\n" + "=" * 80)
interpretation_lines.append("КОНЕЦ ИНТЕРПРЕТАЦИИ")
interpretation_lines.append("=" * 80)

# Сохраняем интерпретацию
interpretation_text = '\n'.join(interpretation_lines)
with open(f'{output_dir}/CLINICAL_INTERPRETATION.txt', 'w', encoding='utf-8') as f:
    f.write(interpretation_text)

print(interpretation_text)
print(f"\n Сохранено: CLINICAL_INTERPRETATION.txt")

# ======================================================================================
# 32. ФИНАЛЬНАЯ СВОДКА ВСЕХ ФАЙЛОВ
# ======================================================================================
print("\n" + "=" * 80)
print("32. СПИСОК ВСЕХ СОЗДАННЫХ ФАЙЛОВ")
print("=" * 80)

import glob

all_files = sorted(glob.glob(f'{output_dir}/*'))

print(f"\n Папка: {output_dir}")
print(f" Всего файлов: {len(all_files)}\n")

print("ДАННЫЕ:")
data_files = [f for f in all_files if f.endswith('.csv') or f.endswith('.json') or f.endswith('.pkl')]
for f in data_files:
    print(f"  • {os.path.basename(f)}")

print("\nГРАФИКИ:")
plot_files = [f for f in all_files if f.endswith('.png')]
for f in plot_files:
    print(f"  • {os.path.basename(f)}")

print("\nОТЧЕТЫ:")
report_files = [f for f in all_files if f.endswith('.txt')]
for f in report_files:
    print(f"  • {os.path.basename(f)}")

# ======================================================================================
# ФИНАЛ
# ======================================================================================
print("\n" + "=" * 80)
print(" АНАЛИЗ ПОЛНОСТЬЮ ЗАВЕРШЕН! ")
print("=" * 80)

print(f"\n ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
print(f"   • Финальная модель: {best_model_name}")
print(f"   • Test Accuracy: {models_comparison.loc[best_model_idx, 'Test Accuracy']:.2%}")
print(f"   • Test F1-Score: {best_model_f1:.4f}")
print(f"   • Количество признаков: {X_train.shape[1]}")
print(f"   • Размер датасета: {len(X_resampled)} примеров (после SMOTE)")

print(f"\n КЛЮЧЕВЫЕ ФАЙЛЫ:")
print(f"   • Модель: FINAL_MODEL.pkl")
print(f"   • Отчет: REPORT.txt")
print(f"   • Интерпретация: CLINICAL_INTERPRETATION.txt")
print(f"   • Confusion Matrix: 09_Confusion_Matrix.png")
print(f"   • Feature Importance: 13_Feature_Importance.png")
if shap_available:
    print(f"   • SHAP анализ: 16_SHAP_Beeswarm.png")
if len(numerical_features) > 0:
    print(f"   • Radar Chart: 21_Radar_Chart.png")

print(f"\n Все результаты сохранены в: {output_dir}/")

print("\n" + "=" * 80)
print("ГОТОВО К ПУБЛИКАЦИИ!")
print("=" * 80)
