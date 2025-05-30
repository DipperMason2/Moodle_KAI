import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # ================== Исправленная загрузка данных ==================
    # Определяем абсолютный путь к папке data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    default_file = os.path.join(data_dir, "predictive_maintenance.csv")

    # Создаем папку data если её нет
    os.makedirs(data_dir, exist_ok=True)

    # Проверяем наличие файла
    if not os.path.exists(default_file):
        st.warning(f"Файл по умолчанию не найден по пути: {default_file}")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Файл успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
            return
    elif os.path.exists(default_file):
        try:
            data = pd.read_csv(default_file)
            st.success(f"Используется файл по умолчанию: {default_file}")
        except Exception as e:
            st.error(f"Ошибка при чтении файла по умолчанию: {e}")
            return
    else:
        st.error(
            "Файл не загружен. Пожалуйста:\n"
            f"1. Загрузите файл вручную выше\n"
            f"2. Или поместите файл 'predictive_maintenance.csv' в папку: {data_dir}"
        )
        return

    # ================== Анализ данных ==================
    st.header("Предварительный анализ данных")
    st.write("Первые 5 строк датасета:")
    st.dataframe(data.head())

    st.write("\nИнформация о датасете:")
    st.write(data.info())

    st.write("\nСтатистика по числовым признакам:")
    st.write(data.describe())

    # ================== Визуализация ==================
    st.header("Визуализация данных")
    fig, ax = plt.subplots(figsize=(10, 6))
    data['Machine failure'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Распределение целевой переменной')
    ax.set_xlabel('Machine failure')
    ax.set_ylabel('Количество')
    st.pyplot(fig)

    # ================== Предобработка ==================
    st.header("Предварительная обработка данных")
    try:
        # Удаляем ненужные столбцы
        cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

        # Кодируем категориальный признак
        if 'Type' in data.columns:
            data['Type'] = LabelEncoder().fit_transform(data['Type'])

        st.success("Предобработка выполнена успешно!")
    except Exception as e:
        st.error(f"Ошибка при предобработке данных: {e}")
        return

    # ================== Моделирование ==================
    st.header("Обучение модели")

    # Разделение данных
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование
    numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    try:
        scaler = StandardScaler()
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    except Exception as e:
        st.error(f"Ошибка при масштабировании: {e}")
        return

    # Выбор модели
    model_choice = st.selectbox("Выберите модель:",
                                ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)

    # Обучение
    if st.button("Обучить модель"):
        try:
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.success("Модель успешно обучена!")

            # Оценка
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            st.subheader("Метрики модели")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            st.write(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

            # Confusion Matrix
            st.subheader("Матрица ошибок")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred),
                        annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            # ROC Curve
            st.subheader("ROC-кривая")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"{model_choice} (AUC = {roc_auc_score(y_test, y_proba):.2f}")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ошибка при обучении модели: {e}")

    # ================== Предсказание ==================
    st.header("Прогнозирование на новых данных")
    with st.form("prediction_form"):
        st.write("Введите параметры оборудования:")

        col1, col2 = st.columns(2)
        with col1:
            type_input = st.selectbox("Тип оборудования", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха [K]", value=300.0)
            process_temp = st.number_input("Температура процесса [K]", value=310.0)
        with col2:
            rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", value=0)

        submit = st.form_submit_button("Предсказать")

        if submit:
            if 'model' in st.session_state:
                model = st.session_state.model
                try:
                    # Подготовка ввода
                    input_data = pd.DataFrame({
                        'Type': [type_input],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotational_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                    })

                    # Преобразование
                    type_mapping = {'L': 0, 'M': 1, 'H': 2}
                    input_data['Type'] = type_mapping[type_input]
                    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                    # Предсказание
                    prediction = model.predict(input_data)[0]
                    proba = model.predict_proba(input_data)[0][1]

                    # Вывод
                    st.subheader("Результат")
                    if prediction == 1:
                        st.error(f"Вероятность отказа: {proba:.1%} ❗")
                    else:
                        st.success(f"Вероятность отказа: {proba:.1%} ✅")

                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {e}")
            else:
                st.warning("Сначала обучите модель!")

            st.warning("Пожалуйста, сначала обучите модель!")


if __name__ == "__main__":
    analysis_and_model_page()