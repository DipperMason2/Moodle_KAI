import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Описание задачи: бинарная классификация для предсказания отказов оборудования.
    - Цель: создать модель и веб-приложение для предиктивного обслуживания.
    ---
    ## Этапы работы
    1. Загрузка и предобработка данных.
    2. Разделение данных на обучающую и тестовую выборки.
    3. Обучение моделей (Random Forest, Logistic Regression).
    4. Оценка моделей.
    5. Создание Streamlit-приложения.
    ---
    ## Результаты
    - Random Forest показал Accuracy = 0.95, ROC-AUC = 0.98.
    - Приложение позволяет загружать данные и делать предсказания.
    ---
    ## Заключение
    - Итоги: создана эффективная модель.
    - Улучшения: использование нейронных сетей.
    """

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

if __name__ == "__main__":
    presentation_page()