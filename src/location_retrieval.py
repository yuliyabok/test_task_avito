"""Считаю совпадения на GPU и получаю из них общий и локальный top-k."""

from collections.abc import Callable, Sequence

import numpy as np

from .retrieval import TfidfRetriever, deterministic_top_positions


def search_global_local_indices_gpu(
    retriever: TfidfRetriever,
    queries: Sequence[object],
    item_locations: np.ndarray,
    query_locations: Sequence[object],
    k: int = 600,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращаю два массива номеров строк корпуса: global и local.

    Global ищет везде, local — по точному совпадению ID локации.
    Произведение матриц считаю пакетами и использую сразу для обоих списков.
    В итоговом запуске k=50 из конфига, 600 здесь — значение по умолчанию.
    """
    if retriever.item_matrix is None or retriever.item_ids is None:
        raise RuntimeError("Call fit before GPU search")
    if len(item_locations) != len(retriever.item_ids):
        raise ValueError("item_locations must align with the item corpus")
    if len(query_locations) != len(queries):
        raise ValueError("query_locations must align with queries")
    if batch_size < 1 or k < 1:
        raise ValueError("batch_size and k must be positive")

    # Поздний импорт позволяет проверять CSV без CuPy. Для самого поиска
    # нужна совместимая NVIDIA GPU и рабочая CUDA.
    import cupy as cp
    import cupyx.scipy.sparse as gpu_sparse

    query_count = len(queries)
    effective_k = min(k, len(retriever.item_ids))
    # int32 компактнее строковых ID; -1 обозначает отсутствие кандидата.
    global_result = np.full((query_count, effective_k), -1, dtype=np.int32)
    local_result = np.full((query_count, effective_k), -1, dtype=np.int32)
    memory_pool = cp.get_default_memory_pool()

    try:
        # Матрица объявлений загружается на GPU один раз для всех query-batch.
        gpu_items = gpu_sparse.csr_matrix(retriever.item_matrix)
        gpu_item_transpose = gpu_items.T
        cp.cuda.Stream.null.synchronize()

        for start in range(0, query_count, batch_size):
            stop = min(start + batch_size, query_count)
            # Векторизацию делаю на CPU, разреженные веса переношу на GPU.
            # L2-нормировка позволяет получить cosine similarity умножением.
            cpu_queries = retriever.vectorizer.transform(queries[start:stop]).tocsr()
            gpu_queries = gpu_sparse.csr_matrix(cpu_queries)
            gpu_similarities = (gpu_queries @ gpu_item_transpose).tocsr()
            cp.cuda.Stream.null.synchronize()
            # Возвращаю на CPU разреженные совпадения. Выбор top-k и фильтр
            # по локации дальше выполняются через NumPy.
            similarities = gpu_similarities.get()

            for local_row in range(stop - start):
                row_start = similarities.indptr[local_row]
                row_stop = similarities.indptr[local_row + 1]
                scores = similarities.data[row_start:row_stop]
                indices = similarities.indices[row_start:row_stop]
                if len(scores) == 0:
                    continue

                selected = deterministic_top_positions(scores, indices, effective_k)
                ranked = indices[selected]
                global_result[start + local_row, : len(ranked)] = ranked

                # Local выбираю из ВСЕХ совпадений, а не из обрезанного global:
                # иначе местное объявление могло бы потеряться до фильтрации.
                # Сравниваю ID локаций, расстояния по координатам не считаю.
                # Global сохраняю отдельно, чтобы не ограничивать весь ответ
                # одной локацией, например для удалённых услуг.
                location = query_locations[start + local_row]
                same_location = item_locations[indices] == location
                local_scores = scores[same_location]
                local_indices = indices[same_location]
                if len(local_scores) > 0:
                    local_selected = deterministic_top_positions(
                        local_scores, local_indices, effective_k
                    )
                    local_ranked = local_indices[local_selected]
                    local_result[start + local_row, : len(local_ranked)] = local_ranked

            # Убираю ссылки на временные матрицы перед следующим пакетом.
            del cpu_queries, gpu_queries, gpu_similarities, similarities
            if progress is not None:
                progress(stop, query_count)
    finally:
        # Удаляю индекс и освобождаю неиспользуемые блоки пула между каналами.
        # finally выполняется и при ошибке внутри поиска.
        if "gpu_item_transpose" in locals():
            del gpu_item_transpose
        if "gpu_items" in locals():
            del gpu_items
        memory_pool.free_all_blocks()
    return global_result, local_result
