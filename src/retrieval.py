"""TF-IDF индексы и объединение кандидатов.

В финальной сборке использую fit и fuse_index_arrays, поиск идёт через GPU.
CPU-методы оставлены для отдельного использования: автоматического
переключения на них при отсутствии GPU в основном скрипте нет.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .text import normalize_text


def deterministic_top_positions(
    scores: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """Сортирует по score, разрешая равенства стабильным индексом item."""
    # Возвращаю позиции внутри scores, а не номера строк исходного корпуса.
    # Если совпадений мало, оставляю все и только упорядочиваю их.
    if len(scores) <= k:
        selected = np.arange(len(scores))
    else:
        # Сначала нахожу границу top-k без полной сортировки.
        # Равные оценки на границе разбираю отдельно по индексу объявления,
        # чтобы результат не зависел от перестановки элементов в partition.
        threshold = np.partition(scores, -k)[-k]
        higher = np.flatnonzero(scores > threshold)
        equal = np.flatnonzero(scores == threshold)
        needed = k - len(higher)
        if needed < len(equal):
            equal_order = np.argsort(indices[equal], kind="stable")[:needed]
            equal = equal[equal_order]
        selected = np.concatenate((higher, equal))
    # В lexsort последний ключ главный: сначала оценка по убыванию,
    # затем индекс по возрастанию. Это стабильно при одном порядке корпуса.
    order = np.lexsort((indices[selected], -scores[selected]))
    return selected[order]


@dataclass
class SearchResult:
    """ID и оценки для CPU-поиска через ближайших соседей."""

    item_ids: list[list[object]]
    scores: list[list[float]]


class TfidfRetriever:
    """Один индекс с собственным словарём и IDF для выбранного текстового поля."""

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
        n_jobs: int = 1,
        preprocessor: Callable[[object], str] | None = None,
    ) -> None:
        # Регистр уже приводит normalize_text, float32 экономит память.
        # sublinear_tf заменяет частоту на 1 + log(tf): многократное повторение
        # одного слова в описании не должно давать слишком большой вес.
        # Оставляю стандартные L2-нормировку и сглаживание IDF из sklearn.
        # В word-режиме стандартный токенизатор берёт токены от двух символов.
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            preprocessor=preprocessor or normalize_text,
            lowercase=False,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self.n_jobs = n_jobs
        self.item_ids: np.ndarray | None = None
        self.item_matrix = None
        self.index: NearestNeighbors | None = None

    def fit(self, item_ids: Sequence[object], texts: Sequence[object]) -> "TfidfRetriever":
        """Строит нормализованную разреженную TF-IDF матрицу корпуса."""
        if len(item_ids) != len(texts):
            raise ValueError("item_ids and texts must have equal length")
        if len(item_ids) == 0:
            raise ValueError("Cannot fit an empty item corpus")
        self.item_ids = np.asarray(item_ids, dtype=object)
        # Учу словарь по объявлениям. Незнакомые слова запроса при transform
        # будут пропущены. CSR хранит только ненулевые веса по строкам.
        self.item_matrix = self.vectorizer.fit_transform(texts).tocsr()
        return self

    def search(self, queries: Sequence[object], k: int = 50) -> SearchResult:
        """CPU-поиск для небольшого корпуса через cosine nearest neighbors.

        Может вернуть соседей с нулевой похожестью; мой разбор равных оценок
        здесь не используется. В итоговой сборке этот метод не вызывается.
        """
        if self.item_matrix is None or self.item_ids is None:
            raise RuntimeError("Call fit before search")
        effective_k = min(k, len(self.item_ids))
        if self.index is None:
            self.index = NearestNeighbors(
                metric="cosine", algorithm="brute", n_jobs=self.n_jobs
            )
            self.index.fit(self.item_matrix)
        query_matrix = self.vectorizer.transform(queries)
        distances, indices = self.index.kneighbors(query_matrix, n_neighbors=effective_k)
        ids = self.item_ids[indices]
        return SearchResult(
            item_ids=[row.tolist() for row in ids],
            scores=[row.tolist() for row in (1.0 - distances)],
        )

    def search_indices_batched(
        self,
        queries: Sequence[object],
        k: int = 50,
        batch_size: int = 8,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Пакетный CPU-поиск по ненулевым совпадениям.

        Скалярное произведение L2-нормированных строк равно косинусной
        похожести. Если совпадений мало, свободные места остаются равными -1.
        """
        if self.item_matrix is None or self.item_ids is None:
            raise RuntimeError("Call fit before search")
        if batch_size < 1 or k < 1:
            raise ValueError("batch_size and k must be positive")

        query_count = len(queries)
        effective_k = min(k, len(self.item_ids))
        result = np.full((query_count, effective_k), -1, dtype=np.int32)
        # Размеры произведения: (запросы × признаки) @ (признаки × объявления).
        # Считаю пакетами, чтобы не хранить совпадения всех запросов одновременно.
        item_transpose = self.item_matrix.T

        for start in range(0, query_count, batch_size):
            stop = min(start + batch_size, query_count)
            query_matrix = self.vectorizer.transform(queries[start:stop])
            similarities = (query_matrix @ item_transpose).tocsr()

            for local_row in range(stop - start):
                # В CSR indptr задаёт границы строки, data — оценки,
                # indices — номера объявлений с ненулевым совпадением.
                row_start = similarities.indptr[local_row]
                row_stop = similarities.indptr[local_row + 1]
                scores = similarities.data[row_start:row_stop]
                indices = similarities.indices[row_start:row_stop]
                if len(scores) == 0:
                    continue
                selected = deterministic_top_positions(scores, indices, effective_k)
                ranked = indices[selected]
                result[start + local_row, : len(ranked)] = ranked

            del query_matrix, similarities
            if progress is not None:
                progress(stop, query_count)
        return result


def merge_ranked_results(
    channels: Sequence[Sequence[Sequence[object]]],
    limit: int | None = None,
) -> list[list[object]]:
    """Чередую списки ID по рангам и убираю повторы.

    В финальной сборке используется аналогичный вариант для массивов ниже.
    """
    if not channels:
        return []
    query_count = len(channels[0])
    if any(len(channel) != query_count for channel in channels):
        raise ValueError("All channels must contain the same number of queries")

    merged: list[list[object]] = []
    for query_idx in range(query_count):
        rows = [channel[query_idx] for channel in channels]
        output: list[object] = []
        seen: set[object] = set()
        max_depth = max((len(row) for row in rows), default=0)
        for rank in range(max_depth):
            for row in rows:
                if rank < len(row) and row[rank] not in seen:
                    seen.add(row[rank])
                    output.append(row[rank])
                    if limit is not None and len(output) >= limit:
                        break
            if limit is not None and len(output) >= limit:
                break
        merged.append(output)
    return merged


def fuse_index_arrays(channels: Sequence[np.ndarray], limit: int) -> np.ndarray:
    """Объединяет compact top-k матрицы round-robin без дубликатов."""
    if not channels:
        return np.empty((0, limit), dtype=np.int32)
    query_count = channels[0].shape[0]
    if any(channel.shape[0] != query_count for channel in channels):
        raise ValueError("All channels must contain the same number of queries")

    # Все каналы должны соответствовать одним запросам и порядку объявлений.
    # В итоговом запуске у массивов также одинаковая ширина.
    output = np.full((query_count, limit), -1, dtype=np.int32)
    for query_idx in range(query_count):
        seen: set[int] = set()
        position = 0
        max_depth = max(channel.shape[1] for channel in channels)
        # Обхожу сначала rank=0 всех каналов, затем rank=1 и так далее.
        # Дубликат пропускаю, он не занимает место. Если совпадений не хватает,
        # строку не дополняю случайными объявлениями: останутся значения -1.
        for rank in range(max_depth):
            for channel in channels:
                candidate = int(channel[query_idx, rank])
                if candidate >= 0 and candidate not in seen:
                    seen.add(candidate)
                    output[query_idx, position] = candidate
                    position += 1
                    if position == limit:
                        break
            if position == limit:
                break
    return output
