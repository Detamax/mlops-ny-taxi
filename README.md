# MLOps: предсказание длительности поездки на такси в Нью-Йорке

Учебный проект, в котором основная ценность — не модель, а инфраструктура вокруг неё:
трекинг экспериментов, реестр моделей, сервис инференса и воспроизводимый запуск всего стека
одной командой.

Модель предсказывает длительность поездки в минутах по данным такси Нью-Йорка (NYC TLC).

## Архитектура

```
┌─────────────┐     регистрация     ┌──────────────┐
│  train.py   │────────────────────▶│    MLflow    │
│ (LightGBM)  │   модели и метрик   │  Tracking +  │
└─────────────┘                     │   Registry   │
                                    └──────┬───────┘
                                           │
                                  models:/taxi-duration@champion
                                           │
                                    ┌──────▼───────┐
                                    │   FastAPI    │
                                    │  /predict    │──▶ HTTP
                                    │  /health     │
                                    └──────────────┘
```

Разделение ролей:

- **train.py**  — обучает модель и регистрирует новую версию в реестре
- **промоушен** — помечает версию алиасом `champion` (пока вручную)
- **FastAPI**   — при старте забирает модель по алиасу и отдаёт предсказания

Сервис не знает ни про ID запуска, ни про версию: он запрашивает `champion`.
Смена алиаса меняет боевую модель без пересборки образа.

## Стек

| Компонент | Назначение |
|---|---|
| LightGBM | градиентный бустинг на табличных данных |
| MLflow 3.16 | трекинг экспериментов, Model Registry |
| FastAPI + Pydantic | сервис инференса и валидация входа |
| Docker Compose | локальный запуск всего стека |
| SQLite | backend store для MLflow (для локальной разработки) |

## Данные

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
yellow taxi, формат Parquet.
файл на котоорой обучалась модель - yellow_tripdata_2025-01.parquet

Предобработка:

- таргет — разница между временем высадки и посадки в минутах
- отсечка аномалий: поездки короче 1 минуты и длиннее 60 отбрасываются
- после фильтрации остаётся ~2.29 млн поездок за месяц

Признаки: `trip_distance`, `passenger_count`, `hour`, `weekday`,
`PULocationID`, `DOLocationID` (последние два — категориальные).

## Результаты

| Модель | Признаки | RMSE (мин) |
|---|---|---|
| LinearRegression (baseline) | базовые | 9.71 |
| LightGBM | базовые | 4.44 |
| LightGBM | + час и день недели | 3.73 |

Baseline на линейной регрессии нужен как точка отсчёта: без него непонятно,
выучила модель структуру данных или просто выдаёт среднее.

Зоны посадки и высадки для линейной регрессии бессмысленны — она воспринимает
номер зоны как величину. LightGBM получает их как категории, отсюда основной
прирост качества.

## Запуск

Требования: Docker, Python 3.13.

### 1. Данные

```bash
mkdir -p data
curl -o data/yellow_tripdata_2025-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
```

### 2. Стек

```bash
docker compose up -d
```

- MLflow UI — http://127.0.0.1:5001
- FastAPI docs — http://127.0.0.1:8001/docs

### 3. Обучение и промоушен

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r mlflow/requirements.txt

MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python3 mlflow/train.py --model lgbm

python3 -c "
from mlflow import MlflowClient
c = MlflowClient(tracking_uri='http://127.0.0.1:5001')
c.set_registered_model_alias('taxi-duration', 'champion', 1)
"

docker compose restart fastapi
```

### 4. Проверка

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_distance": 3.5, "passenger_count": 1,
       "PULocationID": 142, "DOLocationID": 236,
       "hour": 18, "weekday": 2}'
```

```json
{"duration_minutes": 20.41}
```

## API

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | статус сервиса и факт загрузки модели |
| POST | `/predict` | предсказание длительности поездки |
| GET | `/docs` | Swagger UI |

Вход валидируется через Pydantic: расстояние, число пассажиров, ID зон, час и
день недели проверяются на допустимые диапазоны. Некорректный запрос получает
422 и не доходит до модели.

## Структура

```
.
├── data/                   # parquet-файлы (не в git)
├── ml-api/
│   ├── app.py              # FastAPI-сервис
│   ├── Dockerfile
│   └── requirements.txt    # только рантайм-зависимости
├── mlflow/
│   ├── train.py            # обучение и регистрация модели
│   ├── Dockerfile          # надстройка над образом MLflow
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```


## Что дальше

- [ ] автоматический промоушен: сравнение метрики новой версии с текущим
      `champion`, переназначение алиаса только при улучшении
- [ ] CI: линтер, тесты препроцессинга, сборка образа, сканирование Trivy,
      smoke-тест поднятого стека
- [ ] PostgreSQL вместо SQLite и MinIO вместо локальной папки для артефактов
- [ ] Airflow: ежемесячный ретрейн на новых данных TLC
- [ ] мониторинг дрейфа через Evidently, логирование предсказаний
- [ ] ленивая загрузка модели: сервис должен подниматься и отвечать 503 на
      `/predict`, а не падать при старте, если модель недоступна

