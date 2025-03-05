import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("<h1 style='color: black;'>Обзор данных видеокарт NVIDIA и AMD</h1>", unsafe_allow_html=True)

# Функция для установки фонового изображения и стилизации элементов
def set_background_image(image_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url}) no-repeat center center fixed;
            background-size: cover;
        }}
        .stTextInput {{
            background-color: black !important;
            color: white !important;
        }}
        .stSelectbox, .stMultiselect {{
            background-color: black !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 18px !important;
        }}
        .stSelectbox div[role="combobox"] > div:first-child,
        .stMultiselect div[role="combobox"] > div:first-child {{
            background-color: black !important;
            color: white !important;
        }}
        .stButton > button {{
            color: white !important;
            background-color: black !important;
        }}
        .stMarkdown h3, .stMarkdown h2 {{
            color: black !important;
        }}
        .stTextInput label, .stSelectbox label, .stMultiselect label {{
            color: white !important;
        }}
        .uploadedFile {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Установить фон (замени URL на нужный)
set_background_image("_")

# Загрузка файлов (множественный выбор)
uploaded_files = st.file_uploader("Загрузите CSV-файлы с видеокартами", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = []  # Список для хранения загруженных DataFrame
    for file in uploaded_files:
        st.markdown(f"<p style='color: black;'>Файл {file.name} успешно загружен!</p>", unsafe_allow_html=True)
        df = pd.read_csv(file)
        dfs.append(df)

    # Объединение всех загруженных файлов в один DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Поле поиска
    search_query = st.text_input("Введите название видеокарты:")

    if search_query:
        result = combined_df[combined_df["GPU Model"].str.contains(search_query, case=False, na=False)]
        if not result.empty:
            st.write("### Найденные результаты:")
            st.write(result)
        else:
            st.write("Видеокарта не найдена.")

    col1, col2 = st.columns(2)

    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    with col1:
        if st.session_state.show_analysis:
            if st.button("Скрыть Анализ Данных"):
                st.session_state.show_analysis = False
        else:
            if st.button("Показать Анализ Данных"):
                st.session_state.show_analysis = True

    with col2:
        selected_gpus = st.multiselect("Выберите до 5 видеокарт для сравнения:", combined_df["GPU Model"].unique(), max_selections=5)

    if selected_gpus:
        comparison_df = combined_df[combined_df["GPU Model"].isin(selected_gpus)]
        st.write(comparison_df)

    if st.session_state.show_analysis:
        st.write("### Анализ данных")
        st.write("#### Первые 5 строк", combined_df.head())
        st.write("#### Размер данных", combined_df.shape)
        st.write("#### Информация о данных")
        st.text(combined_df.info())
        st.write("#### Описание данных", combined_df.describe())

        # Количество значений
        column_to_count = st.selectbox("Выберите колонку для value_counts()", combined_df.columns)
        st.write(combined_df[column_to_count].value_counts())

        # Сортировка данных
        sort_column = st.selectbox("Выберите колонку для сортировки", combined_df.columns)
        st.write(combined_df.sort_values(by=sort_column))

        # Фильтрация с isin()
        filter_column = st.selectbox("Выберите колонку для фильтрации (isin())", combined_df.columns)
        unique_values = combined_df[filter_column].unique()
        selected_values = st.multiselect("Выберите значения", unique_values)
        st.write(combined_df[combined_df[filter_column].isin(selected_values)])

        # Использование loc и iloc
        st.write("#### Использование loc и iloc")
        st.write("Пример loc (первые 5 строк GPU Model и Price (then)):")
        st.write(combined_df.loc[:5, ["GPU Model", "Price (then)"]])
        st.write("Пример iloc (первые 5 строк первых 3 колонок):")
        st.write(combined_df.iloc[:5, :3])

        # Группировка данных
        st.write("#### Группировка данных")
        group_column = st.selectbox("Выберите колонку для группировки", combined_df.columns)
        grouped_df = combined_df.groupby(group_column).mean()
        st.write(grouped_df)

        # Визуализация данных
        st.write("### Визуализация данных")

        # Boxplot
        st.write("#### Boxplot")
        fig, ax = plt.subplots()
        combined_df.boxplot(column="Price (then)", ax=ax)
        st.pyplot(fig)

        # Scatter plot
        st.write("#### Scatter plot")
        fig, ax = plt.subplots()
        ax.scatter(combined_df["CUDA Cores"], combined_df["Boost Clock (MHz)"], color='blue')
        ax.set_xlabel("CUDA Cores")
        ax.set_ylabel("Boost Clock (MHz)")
        st.pyplot(fig)

        # Histogram
        st.write("#### Histogram")
        fig, ax = plt.subplots()
        combined_df["Memory Size"].hist(bins=10, ax=ax, color='green')
        ax.set_xlabel("Memory Size (GB)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Pie chart
        st.write("#### Pie Chart")
        pie_column = st.selectbox("Выберите колонку для pie chart", combined_df.columns)
        fig, ax = plt.subplots()
        combined_df[pie_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
