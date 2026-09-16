import argparse
import mlflow
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["argparse", "lgbm"], default="lgbm")
args = parser.parse_args()

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("taxi-duration")

df = pd.read_parquet("data/yellow_tripdata_2025-01.parquet")
df["duration"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)]

cat_features = ["PULocationID", "DOLocationID"]
features = ["trip_distance", "passenger_count"] + cat_features
df = df.dropna(subset=features)

df["hour"] = df.tpep_pickup_datetime.dt.hour
df["weekday"] = df.tpep_pickup_datetime.dt.weekday
features = ["trip_distance", "passenger_count", "hour", "weekday"] + cat_features


if args.model == "lgbm":
    for col in cat_features:
        df[col] = df[col].astype("category")

X_train, X_val, y_train, y_val = train_test_split(
    df[features], df["duration"], test_size=0.2, random_state=42
)

with mlflow.start_run(run_name=args.model):
    if args.model == "argparse":
        model = LinearRegression()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, name="model")
    else:
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1)
        model.fit(X_train, y_train, categorical_feature=cat_features)
        mlflow.lightgbm.log_model(model, name="model")

    rmse = root_mean_squared_error(y_val, model.predict(X_val))
    mlflow.log_param("model_type", args.model)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_metric("rmse", rmse)
    print(f"RMSE: {rmse:.3f}")