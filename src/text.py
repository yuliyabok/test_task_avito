"""Нормализация текста и отдельный необязательный лемматизатор.

В финальной сборке используется normalize_text. tokenize и
RussianLemmaNormalizer к трём итоговым индексам не подключены.
"""

from functools import lru_cache
import math
import re
import unicodedata


TOKEN_PATTERN = re.compile(r"[а-яa-z0-9]+")
CYRILLIC_WORD_PATTERN = re.compile(r"[а-я]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Нормализует Unicode, регистр, букву ё и лишние пробелы."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    # NFKC сводит совместимые Unicode-формы к единому виду.
    # Регистр и различие «ё»/«е» для поиска услуги не нужны. Пунктуацию
    # здесь не удаляю: дальше её обработка зависит от анализатора.
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.lower().replace("ё", "е")
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def tokenize(value: object) -> list[str]:
    """Выделяет русские, латинские и числовые токены."""
    return TOKEN_PATTERN.findall(normalize_text(value))


class RussianLemmaNormalizer:
    """Привожу русские слова к начальной форме, остальные токены оставляю.

    Кэш позволяет не разбирать одно слово заново в каждом объявлении.
    Класс не используется в финальной сборке. Для отдельного запуска
    нужен pymorphy3, которого нет в текущем requirements.txt.
    """

    def __init__(self, cache_size: int = 500_000) -> None:
        try:
            import pymorphy3
        except ImportError as error:
            raise RuntimeError(
                "Russian lemmatization requires pymorphy3; install requirements.txt"
            ) from error
        self._morph = pymorphy3.MorphAnalyzer()
        self._cached_lemma = lru_cache(maxsize=cache_size)(self._lemma_uncached)

    def _lemma_uncached(self, token: str) -> str:
        if CYRILLIC_WORD_PATTERN.fullmatch(token) is None:
            return token
        # Выбираю первый разбор без контекста: для неоднозначного слова
        # начальная форма может оказаться неверной.
        return self._morph.parse(token)[0].normal_form.replace("ё", "е")

    def __call__(self, value: object) -> str:
        return " ".join(self._cached_lemma(token) for token in tokenize(value))

    def cache_info(self):
        """Показываю размер кэша и попадания в него для диагностики."""
        return self._cached_lemma.cache_info()
