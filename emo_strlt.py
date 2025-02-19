import streamlit as st
import pandas as pd
import re
import pymorphy2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

st.title("Анализ заголовков")

# сбоку
st.sidebar.header("Загрузка файлов и настройки")

# 1
uploaded_files = st.sidebar.file_uploader("Загрузите CSV‑файлы", type=["csv"], accept_multiple_files=True)
# 2
if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = {}
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["dataframes"]:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state["dataframes"][uploaded_file.name] = df
            except Exception as e:
                st.sidebar.error(f"Ошибка при чтении файла {uploaded_file.name}: {e}")

# 3
if st.session_state["dataframes"]:
    available_files = list(st.session_state["dataframes"].keys())
    st.sidebar.write("Загруженные файлы:")
    for f in available_files:
        st.sidebar.write(f)
else:
    st.info("Загрузите CSV‑файлы для анализа.")
# выбор файлов
selected_files = st.sidebar.multiselect("Выберите файлы для анализа", options=st.session_state["dataframes"].keys(),
                                        default=list(st.session_state["dataframes"].keys()))
# выбор режима
analysis_mode = st.sidebar.radio("Режим анализа:", ("По отдельности", "Объединить файлы"))
# выбор подхода
method = st.sidebar.selectbox("Выберите подход:", ("TF‑IDF", "CountVectorizer"))
# выбор диапазона n‑грамм
ngram_choice = st.sidebar.radio("Выберите диапазон n‑грамм:",
                                ("Все n‑граммы (1,3)", "Только биграммы и триграммы (2,3)"))
if ngram_choice == "Все n‑граммы (1,3)":
    ngram_range = (1, 3)
else:
    ngram_range = (2, 3)
# удаляем
if st.sidebar.button("Удалить все загруженные датафреймы"):
    if "dataframes" in st.session_state:
        st.session_state.pop("dataframes")
        st.sidebar.success("Все датафреймы удалены!")
    else:
        st.sidebar.info("Нет загруженных датафреймов.")

custom_stopwords = set([
    "в", "на", "с", "по", "от", "до", "из", "у", "о", "об",
    "под", "при", "к", "за", "через", "для", "без", "над", "про",
    "между", "сквозь", "вокруг", "это", "около"
])
russian_stopwords = set(stopwords.words("russian"))
stop_words = custom_stopwords.union(russian_stopwords)
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    processed_words = []
    for word in words:
        if not re.match(r'^[а-яё]+$', word):
            continue
        lemma = morph.parse(word)[0].normal_form
        processed_words.append(lemma)
    return " ".join(processed_words)

def filter_unigram(term, stop_words_set):
    words = term.split()
    if len(words) == 1 and words[0] in stop_words_set:
        return False
    return True

def analyze_dataframe(df, filename):
    if 'name' not in df.columns:
        st.error(f"В файле {filename} отсутствует столбец 'name'.")
        return None

    df['processed_name'] = df['name'].apply(preprocess_text)
    if method == "TF‑IDF":
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range)

    matrix = vectorizer.fit_transform(df['processed_name'])
    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    term_dict = dict(zip(feature_names, scores))
    sorted_terms = sorted(term_dict.items(), key=lambda x: x[1], reverse=True)

    if ngram_range[0] == 1:
        sorted_terms = [(term, score) for term, score in sorted_terms if filter_unigram(term, stop_words)]

    return sorted_terms

results = {}

if selected_files:
    if analysis_mode == "По отдельности":
        # анализ каждого выбранного файла отдельно
        for filename in selected_files:
            df = st.session_state["dataframes"][filename]
            sorted_terms = analyze_dataframe(df, filename)
            if sorted_terms is not None:
                results[filename] = sorted_terms
                st.subheader(f"Файл: {filename}")
                st.write(f"Топ 100 терминов для файла **{filename}**:")
                for term, score in sorted_terms[:100]:
                    if method == "TF‑IDF":
                        st.write(f"{term}: {score:.4f}")
                    else:
                        st.write(f"{term}: {int(score)}")
                results_df = pd.DataFrame(sorted_terms, columns=["Термин", "Оценка"])
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Скачать результаты для файла {filename}",
                    data=csv,
                    file_name=f"{filename}_results.csv",
                    mime="text/csv",
                )
                if st.checkbox(f"Показать всю таблицу терминов для файла {filename}", key=filename):
                    st.dataframe(results_df)
    else:
        combined_dfs = []
        for filename in selected_files:
            df = st.session_state["dataframes"][filename].copy()
            if "id" in df.columns:
                df["id"] = filename + "_" + df["id"].astype(str)
            combined_dfs.append(df)
        combined_df = pd.concat(combined_dfs, ignore_index=True)

        st.subheader("Объединённый DataFrame для анализа")
        st.write(f"Всего строк: {combined_df.shape[0]}")

        sorted_terms = analyze_dataframe(combined_df, "Объединённый файл")
        if sorted_terms is not None:
            results["Объединённый файл"] = sorted_terms
            st.write("Топ 100 терминов для объединённого файла:")
            for term, score in sorted_terms[:100]:
                if method == "TF‑IDF":
                    st.write(f"{term}: {score:.4f}")
                else:
                    st.write(f"{term}: {int(score)}")

            results_df = pd.DataFrame(sorted_terms, columns=["Термин", "Оценка"])
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать результаты для объединённого файла",
                data=csv,
                file_name="combined_results.csv",
                mime="text/csv",
            )
            if st.checkbox("Показать всю таблицу терминов для объединённого файла"):
                st.dataframe(results_df)
else:
    st.info("Выберите файлы для анализа в боковой панели.")
