from pathlib import Path
from typing import List, Optional

import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Данные пассажиров Титаника",
    page_icon="🚢",
    layout="centered",
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "titanic_train.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# Возвращает отсортированный список вариантов для выбора пункта посадки
def prepare_embarked_options(df: pd.DataFrame) -> List[str]:
    codes = df["Embarked"].dropna().astype(str).unique().tolist()
    codes.sort()
    return ["Все порты"] + codes


# Фильтрует DataFrame по значению Embarked. Возвращает копию при фильтрации
def filter_by_embarked(df: pd.DataFrame, embarked: Optional[str]) -> pd.DataFrame:
    if not embarked or embarked == "Все порты":
        return df.copy()
    return df[df["Embarked"] == embarked]


# Возвращает таблицу с количеством спасённых и погибших пассажиров
def summarize_survival(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["Survived"]
        .value_counts()
        .reindex([1, 0], fill_value=0)
        .rename(index={1: "Спасены", 0: "Погибли"})
    )
    return pd.DataFrame(
        {
            "Статус": counts.index,
            "Количество пассажиров": counts.values,
        }
    )


df = load_data()

st.title("Данные пассажиров Титаника")
st.write("Выберите пункт посадки")

embarked_options = prepare_embarked_options(df)
selected_embarked = st.selectbox("Пункт посадки:", embarked_options)

filtered_df = filter_by_embarked(df, selected_embarked)
subtitle = (
    "по всем портам посадки"
    if selected_embarked == "Все порты"
    else f"для порта {selected_embarked}"
)

st.subheader(f"Число пассажиров {subtitle}")

result_table = summarize_survival(filtered_df)

st.table(result_table)
