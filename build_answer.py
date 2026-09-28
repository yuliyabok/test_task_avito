#!/usr/bin/env python3
"""Собираю до 50 объявлений для каждого запроса и проверяю готовый CSV.

Три текстовых индекса дают шесть списков кандидатов: общий и локальный
для каждого представления текста. Затем объединяю их без повторов.
Режим --validate-only проверяет существующий ответ без запуска GPU.
"""

import os

# Ограничиваю потоки до импорта NumPy: численные библиотеки могут создавать
# свои пулы одновременно. Уже заданные настройки окружения не перезаписываю.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")
# Ограничиваю число арен аллокатора, чтобы сдержать расход оперативной памяти.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Пути считаю от самого скрипта, чтобы запуск не зависел от текущей папки.
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.location_retrieval import search_global_local_indices_gpu  # noqa: E402
from src.retrieval import TfidfRetriever, fuse_index_arrays  # noqa: E402
from src.validation import build_answer_frame, validate_answer_frame  # noqa: E402


# При слиянии беру первое объявление каждого канала, затем второе и т.д.
# Порядок здесь важен: при достижении лимита более ранний канал может занять
# последнее место. Это фиксированное правило, а не обученные веса.
CHANNEL_ORDER = [
    "word_title_global",
    "word_description_global",
    "word_title_local",
    "word_description_local",
    "char_title_global",
    "char_title_local",
]


def parse_args() -> argparse.Namespace:
    """Читает пути и режим запуска из командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    # По умолчанию parquet лежат рядом с final_solution; путь можно переопределить.
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_ROOT.parent,
        help="Directory containing benchmark_queries.parquet and benchmark_items.parquet",
    )
    parser.add_argument(
        "--config", type=Path, default=PACKAGE_ROOT / "config.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=PACKAGE_ROOT / "answer.csv"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "submission_metrics.json",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing output without rebuilding retrieval indexes",
    )
    return parser.parse_args()


def make_progress(label: str, every: int):
    """Создаёт компактный callback для вывода прогресса."""
    # Печатаю прогресс через заданный интервал и обязательно в конце.
    # Иначе короткие пакеты запросов создавали бы слишком много сообщений.
    last_report = 0

    def report(done: int, total: int) -> None:
        nonlocal last_report
        if done == total or done - last_report >= every:
            print(f"[{label}] queries: {done}/{total}", flush=True)
            last_report = done

    return report


def write_metrics(path: Path, metrics: dict) -> None:
    """Сохраняет метрики запуска в читаемый JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_benchmark(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает benchmark и проверяет уникальность идентификаторов."""
    # train.parquet в итоговом запуске не нужен: словарь и IDF строю по корпусу.
    # ID нужны для связи строк с результатом, их содержимое не является признаком.
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")
    if queries["query_id"].duplicated().any():
        raise RuntimeError("benchmark query_id values are not unique")
    if items["item_id"].duplicated().any():
        raise RuntimeError("benchmark item_id values are not unique")
    return queries, items


def validate_output(
    output_path: Path,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    final_k: int,
) -> dict:
    """Повторно читает CSV и строго проверяет формат итогового ответа."""
    # ID читаю строками, чтобы сохранить ведущие нули и регистр.
    # Пустой ответ оставляю строкой, а не превращаю в NaN.
    answer = pd.read_csv(
        output_path,
        dtype={"query_id": str, "answer": str},
        keep_default_na=False,
    )
    allowed_item_ids = set(items["item_id"].astype(str))
    validate_answer_frame(
        answer,
        queries["query_id"].astype(str).tolist(),
        allowed_item_ids,
        max_items=final_k,
    )
    # Это проверка заполненности и формата, не Recall@50.
    # SHA-256 позволяет связать отчёт с конкретным сохранённым CSV.
    counts = answer["answer"].map(lambda value: len(value.split()) if value else 0)
    return {
        "answer_rows": len(answer),
        "minimum_predictions": int(counts.min()),
        "maximum_predictions": int(counts.max()),
        "mean_predictions": float(counts.mean()),
        "empty_answers": int(np.count_nonzero(counts == 0)),
        "duplicate_query_ids": int(answer["query_id"].duplicated().sum()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "strict_validation_passed": True,
    }


def main() -> None:
    """Строит шесть retrieval-каналов и формирует валидный top-50."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    retrieval_config = config["retrieval"]
    char_config = config["character_retrieval"]
    submission = config["submission"]
    depth = int(submission["candidates_per_channel"])
    final_k = int(submission["final_k"])

    print("Loading benchmark queries and item corpus", flush=True)
    queries, items = load_benchmark(args.data_dir)
    print(f"Prepared {len(queries)} queries and {len(items)} items", flush=True)

    if args.validate_only:
        validation = validate_output(args.output, queries, items, final_k)
        print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
        return

    # CuPy нужен только для полного построения, но не для validate-only.
    import cupy as cp

    # Во всех каналах сохраняю один порядок корпуса: поиск вернёт номера строк,
    # а в item_id я переведу их только после объединения кандидатов.
    item_ids = items["item_id"].to_numpy(dtype=object)
    item_locations = items["item_location_id"].to_numpy()
    query_texts = queries["search_query"].fillna("").astype(str).tolist()
    query_locations = queries["search_location_id"].to_numpy()
    # Заголовок даёт короткое название услуги, описание добавляет детали.
    # Индексирую поля отдельно, чтобы длинное описание не меняло веса заголовка.
    # Символьные фрагменты помогают при частичном совпадении слов, например
    # разных окончаниях или опечатках. Синонимы без общих букв они не распознают.
    # Каждое из трёх представлений даст общий и локальный список.
    specifications = [
        (
            "word_title",
            "item_title_raw",
            "word",
            tuple(retrieval_config["ngram_range"]),
            retrieval_config["min_df"],
        ),
        (
            "word_description",
            "item_description_raw",
            "word",
            tuple(retrieval_config["ngram_range"]),
            retrieval_config["min_df"],
        ),
        (
            "char_title",
            char_config["field"],
            "char_wb",
            tuple(char_config["ngram_range"]),
            char_config["min_df"],
        ),
    ]
    channels: dict[str, np.ndarray] = {}
    channel_metrics: dict[str, dict] = {}
    started = time.perf_counter()

    # Индексы обрабатываются последовательно, чтобы ограничить расход памяти.
    for name, field, analyzer, ngram_range, min_df in specifications:
        channel_started = time.perf_counter()
        print(f"[{name}] fitting benchmark item texts", flush=True)
        retriever = TfidfRetriever(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            n_jobs=1,
        ).fit(item_ids, items[field].fillna(""))
        # fit здесь строит словарь и IDF, а не обучает модель на выборе пользователя.
        matrix_shape = retriever.item_matrix.shape
        matrix_nnz = int(retriever.item_matrix.nnz)
        search_started = time.perf_counter()
        global_predictions, local_predictions = search_global_local_indices_gpu(
            retriever,
            query_texts,
            item_locations=item_locations,
            query_locations=query_locations,
            k=depth,
            batch_size=int(submission["gpu_batch_size"]),
            progress=make_progress(name, int(submission["progress_every"])),
        )
        channels[f"{name}_global"] = global_predictions
        channels[f"{name}_local"] = local_predictions
        # Считаю реальные кандидаты: -1 — пустое место. nnz показывает число
        # ненулевых весов в индексе и помогает оценить его размер.
        channel_metrics[name] = {
            "matrix_features": matrix_shape[1],
            "matrix_nnz": matrix_nnz,
            "global_mean_candidates": float(
                np.mean(np.count_nonzero(global_predictions >= 0, axis=1))
            ),
            "local_mean_candidates": float(
                np.mean(np.count_nonzero(local_predictions >= 0, axis=1))
            ),
            "search_seconds": time.perf_counter() - search_started,
            "elapsed_seconds": time.perf_counter() - channel_started,
        }
        del retriever
        gc.collect()

    # Чередую каналы по позициям, не складывая несопоставимые TF-IDF оценки.
    # Объявление может встретиться в нескольких списках, но в ответ войдёт один раз.
    fused = fuse_index_arrays([channels[name] for name in CHANNEL_ORDER], final_k)
    # Сначала убираю -1: иначе NumPy подставил бы последнее объявление корпуса.
    predictions = [
        item_ids[row[row >= 0]].astype(str).tolist() for row in fused
    ]
    answer = build_answer_frame(
        queries["query_id"].astype(str).tolist(),
        predictions,
        set(items["item_id"].astype(str)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(args.output, index=False, encoding="utf-8")

    # Проверяем уже сериализованный файл, а не только DataFrame в памяти.
    validation = validate_output(args.output, queries, items, final_k)
    metrics = {
        "experiment": "benchmark_six_channel_round_robin_submission",
        "backend": "cupy-cusparse",
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        # Seed сохраняю как настройку запуска; случайных операций здесь нет.
        "seed": config["seed"],
        "benchmark_queries": len(queries),
        "benchmark_items": len(items),
        "per_channel_depth": depth,
        "final_k": final_k,
        "channel_order": CHANNEL_ORDER,
        "channels": channel_metrics,
        **validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_metrics(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Answer saved to {args.output}", flush=True)
    print(f"Metrics saved to {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
