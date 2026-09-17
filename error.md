# Ошибка при запуске docker-compose

## Симптом

```
mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=taxi-duration not found
fastapi-1 | ERROR: Application startup failed. Exiting.
```

FastAPI падает при старте, перезапускается в цикле, после чего MLflow тоже завершается с кодом 137 (OOM/kill).

---

## Корневая причина

**Используется не та база данных MLflow.**

Обучение проводилось локально — MLflow-сервер хранил модели в файле **`./mlflow.db`** (корень проекта). В этой БД зарегистрирована модель `taxi-duration` с алиасом `champion`.

Docker-compose монтирует том `./mlruns:/mlruns` и запускает MLflow с параметром:
```
--backend-store-uri sqlite:////mlruns/mlflow.db
```

Файл `./mlruns/mlflow.db` — это **другая, пустая база данных** (0 зарегистрированных моделей, 0 версий).

### Сравнение двух баз данных

| Файл | Модели | Алиасы |
|---|---|---|
| `./mlflow.db` (корень) | `taxi-duration` v1, статус READY | `champion` |
| `./mlruns/mlflow.db` (используется в Docker) | — | — |

### Цепочка ошибки

1. MLflow-контейнер стартует с пустой `mlruns/mlflow.db`
2. FastAPI при старте вызывает `mlflow.pyfunc.load_model("models:/taxi-duration@champion")`
3. MLflow возвращает 404 — модель не найдена в пустой БД
4. FastAPI падает → перезапускается (`restart: unless-stopped`) → снова падает → цикл

---

## Дополнительные проблемы

- **Рассинхрон путей артефактов**: модель была зарегистрирована по URI `mlflow-artifacts:/1/models/m-2f23c43dec0448f1b3bc7d6a2a6c6911/artifacts`, но новый MLflow-сервер не знает об этом маппинге.
- **Различие версий Python**: MLflow-контейнер — Python 3.13, FastAPI-контейнер — Python 3.12.

---

## Решение

### Вариант 1 (быстрый) — скопировать корректную БД

```bash
cp mlflow.db mlruns/mlflow.db
```

Затем пересобрать и запустить:
```bash
docker-compose up --build
```

### Вариант 2 (правильный) — изменить путь backend store в docker-compose.yml

Добавить корневой `mlflow.db` в том и указать правильный путь:

```yaml
services:
  mlflow:
    build: ./mlflow
    ports:
      - "5001:5000"
    volumes:
      - ./mlruns:/mlruns
      - ./mlartifacts:/mlartifacts
      - ./mlflow.db:/mlflow.db        # <- добавить
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////mlflow.db   # <- изменить путь
      --artifacts-destination /mlartifacts
      --allowed-hosts "*"
```

---

## Почему это произошло

При локальном обучении MLflow-сервер запускался с одной конфигурацией (используя `mlflow.db` в корне), а при сборке docker-compose был указан другой путь к БД — `mlruns/mlflow.db`. Артефакты модели физически существуют в `./mlartifacts/`, но метаданные о регистрации (имя модели, алиас `champion`) хранятся только в оригинальной `./mlflow.db`.
