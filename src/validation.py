"""Проверяю формат и состав ответа; релевантность здесь не оценивается."""

from collections.abc import Sequence
import re

import pandas as pd


# У item_id ровно 16 строчных шестнадцатеричных символов. У query_id
# проверяю длину и совпадение с исходным ID с сохранением регистра.
ITEM_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def validate_predictions(
    query_ids: Sequence[object],
    predictions: Sequence[Sequence[object]],
    allowed_item_ids: set[object],
    max_items: int = 50,
) -> None:
    """Проверяет длину, уникальность и принадлежность item_id корпусу."""
    if len(query_ids) != len(predictions):
        raise ValueError("Every query must have exactly one prediction row")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_id values must be unique")
    # Пустой список допустим по условию, повторы и превышение лимита — нет.
    for query_id, items in zip(query_ids, predictions):
        if len(items) > max_items:
            raise ValueError(f"Query {query_id} has more than {max_items} items")
        if len(set(items)) != len(items):
            raise ValueError(f"Query {query_id} contains duplicate item IDs")
        # Одного правильного формата ID мало: объявление должно быть в корпусе.
        unknown = set(items) - allowed_item_ids
        if unknown:
            raise ValueError(f"Query {query_id} contains items outside benchmark corpus")


def build_answer_frame(
    query_ids: Sequence[object],
    predictions: Sequence[Sequence[object]],
    allowed_item_ids: set[object],
) -> pd.DataFrame:
    """Сериализует списки рекомендаций в формат query_id,answer."""
    # Проверяю списки ещё до склейки в строки и записи CSV.
    validate_predictions(query_ids, predictions, allowed_item_ids)
    return pd.DataFrame(
        {
            "query_id": query_ids,
            "answer": [" ".join(map(str, items)) for items in predictions],
        }
    )


def validate_answer_frame(
    answer: pd.DataFrame,
    expected_query_ids: Sequence[object],
    allowed_item_ids: set[object],
    max_items: int = 50,
) -> None:
    """Строго проверяет контракт готового benchmark CSV."""
    # Лишняя колонка индекса тоже ошибка. Порядок строк запросов свободный,
    # но набор query_id должен точно совпадать с исходным benchmark.
    if answer.columns.tolist() != ["query_id", "answer"]:
        raise ValueError("Answer must contain exactly query_id,answer columns")
    query_ids = answer["query_id"].astype(str).tolist()
    expected = list(map(str, expected_query_ids))
    if len(query_ids) != len(expected) or set(query_ids) != set(expected):
        raise ValueError("Answer query IDs must exactly match benchmark queries")
    if any(len(query_id) != 16 for query_id in query_ids):
        raise ValueError("Every query_id must contain exactly 16 characters")

    # Разбираю сохранённые строки обратно на ID и проверяю списки повторно.
    # Уникальность query_id ниже дополняет сравнение множеств выше.
    predictions: list[list[str]] = []
    for value in answer["answer"].fillna("").astype(str):
        items = value.split() if value else []
        invalid_format = [item for item in items if ITEM_ID_PATTERN.fullmatch(item) is None]
        if invalid_format:
            raise ValueError(f"Invalid item_id format: {invalid_format[:3]}")
        predictions.append(items)
    validate_predictions(query_ids, predictions, allowed_item_ids, max_items=max_items)
