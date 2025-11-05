import pandas as pd
import pytest

from app import prepare_embarked_options, filter_by_embarked, summarize_survival


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 1],
            "Embarked": ["S", "C", "S", "Q", None],
        }
    )


# Проверяем, что коды отсортированы и без пропусков
def test_prepare_embarked_options_sorted_with_all_port(sample_df):
    options = prepare_embarked_options(sample_df)
    assert options[0] == "Все порты"
    assert options[1:] == ["C", "Q", "S"]


# Проверяем, что функция возвратила независимую копию данных (оригинал не изменится)
def test_filter_by_embarked_paths(sample_df):
    filtered = filter_by_embarked(sample_df, "Все порты")
    assert len(filtered) == len(sample_df)
    filtered.loc[:, "Survived"] = 99
    assert sample_df["Survived"].iloc[0] == 0


# Проверяем реакцию на несуществующий порт
def test_filter_by_embarked_unknown_port(sample_df):
    filtered = filter_by_embarked(sample_df, "X")
    assert filtered.empty


# Проверяет корректность формирования итоговой сводки
def test_summarize_survival_counts_order_and_values(sample_df):
    df = filter_by_embarked(sample_df, "S")
    summary = summarize_survival(df)
    assert summary["Статус"].tolist() == ["Спасены", "Погибли"]
    assert summary["Количество пассажиров"].tolist() == [1, 1]
