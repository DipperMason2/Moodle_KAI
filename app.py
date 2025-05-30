import streamlit as st

# Определение страниц
pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
]

# Настройка навигации
st.navigation(pages, position="sidebar", expanded=True)