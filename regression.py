# regression_full.py
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ==========================
# AYARLAR
# ==========================
DATA_PATH = "dataset/chicago_crime1.csv"   # csv veya parquet
DATE_COL = "Date"
LAT_COL  = "Latitude"
LON_COL  = "Longitude"

DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # "10/17/2025 12:00:00 AM"

CELL_SIZE_M = 500
UTM_EPSG = "EPSG:32616"  # Chicago UTM zone 16N

MIN_YEAR = 2014  # None yaparsan tüm yıllar

# Aktif hücre filtresi (weekly üzerinden yapılacak)
MIN_TOTAL_CRIMES_PER_CELL = 50  # 20/50/100 deneyebilirsin

OUT_PANEL_PATH = "panel_weekly.parquet"
OUT_PRED_PATH  = "predictions_with_geo.parquet"
# ==========================


def read_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={LAT_COL: "lat", LON_COL: "lon", DATE_COL: "date"})


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lat", "lon", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    df = df.dropna(subset=["date"])

    if MIN_YEAR is not None:
        df = df[df["date"].dt.year >= MIN_YEAR].copy()

    # Chicago bounding box
    df = df[df["lat"].between(41.60, 42.10) & df["lon"].between(-87.95, -87.45)].copy()

    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    df = df.reset_index(drop=True)
    return df


def add_utm_and_grid(df: pd.DataFrame):
    transformer = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())

    df = df.copy()
    df["x"] = x
    df["y"] = y

    x0 = float(df["x"].min())
    y0 = float(df["y"].min())

    df["gx"] = np.floor((df["x"] - x0) / CELL_SIZE_M).astype("int32")
    df["gy"] = np.floor((df["y"] - y0) / CELL_SIZE_M).astype("int32")
    df["cell_id"] = df["gx"].astype(str) + "_" + df["gy"].astype(str)
    return df, x0, y0


def add_week_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ Week start'ı GERÇEKTEN Pazartesi yapmak için:
    - W-SUN: hafta Pazar biter
    - start_time: Pazartesi olur
    """
    df = df.copy()
    df["week_start"] = df["date"].dt.to_period("W-SUN").dt.start_time
    return df


def make_weekly_counts(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["cell_id", "gx", "gy", "week_start"])
          .size()
          .reset_index(name="crime_count")
    )
    # Tipleri sabitle (merge güvenliği)
    weekly["cell_id"] = weekly["cell_id"].astype(str)
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    return weekly


def select_active_cells_from_weekly(weekly: pd.DataFrame, min_total: int):
    totals = weekly.groupby("cell_id")["crime_count"].sum()
    keep = totals[totals >= min_total].index
    print("ACTIVE from weekly:", len(keep), "cells (threshold:", min_total, ")")
    return keep


def make_full_panel(weekly: pd.DataFrame) -> pd.DataFrame:
    # sadece aktif hücreleri kapsayan hücre listesi
    all_cells = weekly[["cell_id", "gx", "gy"]].drop_duplicates().copy()
    all_cells["cell_id"] = all_cells["cell_id"].astype(str)

    # ✅ En garanti: haftaları weekly'nin unique değerlerinden al
    all_weeks = pd.DatetimeIndex(np.sort(weekly["week_start"].unique()))

    panel = all_cells.assign(key=1).merge(
        pd.DataFrame({"week_start": all_weeks, "key": 1}),
        on="key"
    ).drop(columns="key")

    panel["cell_id"] = panel["cell_id"].astype(str)
    panel["week_start"] = pd.to_datetime(panel["week_start"])

    panel = panel.merge(
        weekly[["cell_id", "week_start", "crime_count"]],
        on=["cell_id", "week_start"],
        how="left"
    )

    panel["crime_count"] = panel["crime_count"].fillna(0).astype("int32")
    panel = panel.sort_values(["cell_id", "week_start"]).reset_index(drop=True)
    return panel


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["month"] = panel["week_start"].dt.month.astype("int8")
    panel["weekofyear"] = panel["week_start"].dt.isocalendar().week.astype("int16")
    return panel


def add_lag_roll_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["cell_id", "week_start"]).copy()
    g = panel.groupby("cell_id")["crime_count"]

    for L in [1, 2, 4, 8, 12]:
        panel[f"lag_{L}"] = g.shift(L)

    panel["roll_mean_4"]  = g.shift(1).rolling(4).mean()
    panel["roll_mean_12"] = g.shift(1).rolling(12).mean()
    panel["roll_mean_24"] = g.shift(1).rolling(24).mean()
    panel["roll_std_12"]  = g.shift(1).rolling(12).std()

    cols = [c for c in panel.columns if c.startswith("lag_") or c.startswith("roll_")]
    panel[cols] = panel[cols].fillna(0)
    return panel


def add_neighbor_mean(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    base = panel[["gx", "gy", "week_start", "crime_count"]].copy()
    base = base.astype({"gx": "int32", "gy": "int32"})

    nbr = panel[["gx", "gy", "week_start"]].copy()

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            tmp = base.copy()
            tmp["gx"] = tmp["gx"] - dx
            tmp["gy"] = tmp["gy"] - dy
            tmp = tmp.rename(columns={"crime_count": f"nbr_{dx}_{dy}"})
            nbr = nbr.merge(tmp, on=["gx", "gy", "week_start"], how="left")

    nbr_cols = [c for c in nbr.columns if c.startswith("nbr_")]
    nbr[nbr_cols] = nbr[nbr_cols].fillna(0)
    panel["neighbor_mean"] = nbr[nbr_cols].mean(axis=1).astype("float32")
    return panel


def time_split(panel: pd.DataFrame):
    weeks = np.sort(panel["week_start"].unique())
    n = len(weeks)
    if n < 30:
        raise RuntimeError(f"Çok az hafta var ({n}). Split yapamayız.")

    train_end = weeks[max(1, int(n * 0.80) - 1)]
    val_end   = weeks[max(2, int(n * 0.90) - 1)]

    train = panel[panel["week_start"] <= train_end].copy()
    val   = panel[(panel["week_start"] > train_end) & (panel["week_start"] <= val_end)].copy()
    test  = panel[panel["week_start"] > val_end].copy()

    print("=== TIME SPLIT ===")
    print("  weeks:", n)
    print("  train_end:", train_end, "val_end:", val_end, "last:", weeks[-1])
    print("  rows:", len(train), len(val), len(test))
    assert len(val) > 0 and len(test) > 0, "Val/Test boş!"
    return train, val, test


def train_model(train, val, feature_cols):
    X_train, y_train = train[feature_cols], train["crime_count"]
    X_val, y_val     = val[feature_cols], val["crime_count"]

    model = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def evaluate(model, test, feature_cols):
    X_test = test[feature_cols]
    y_test = test["crime_count"].to_numpy()

    pred = model.predict(X_test)

    print("=== TEST DEBUG ===")
    print("y_test describe:\n", pd.Series(y_test).describe())
    print("y_test zero ratio:", float((y_test == 0).mean()))
    print("y_test nonzero:", int((y_test > 0).sum()))
    print("pred describe:\n", pd.Series(pred).describe())

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    baseline = X_test["lag_1"].to_numpy()
    b_mae = mean_absolute_error(y_test, baseline)
    b_rmse = np.sqrt(mean_squared_error(y_test, baseline))

    print(f"✅ MODEL  MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    print(f"✅ BASELINE(lag_1) MAE: {b_mae:.4f}  RMSE: {b_rmse:.4f}")
    return pred


def save_predictions_geo(test, pred, x0, y0):
    out = test[["cell_id", "gx", "gy", "week_start"]].copy()
    out["pred_count"] = pred

    out["x_center"] = x0 + (out["gx"] + 0.5) * CELL_SIZE_M
    out["y_center"] = y0 + (out["gy"] + 0.5) * CELL_SIZE_M

    inv = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(out["x_center"].to_numpy(), out["y_center"].to_numpy())
    out["lon_center"] = lon
    out["lat_center"] = lat

    out.to_parquet(OUT_PRED_PATH, index=False)
    print(f"✅ Saved: {OUT_PRED_PATH}")


def main():
    print("=== 1) READ ===")
    df = read_data(DATA_PATH)
    print("Ham satır:", len(df))

    print("=== 2) CLEAN ===")
    df = standardize_columns(df)
    df = clean_data(df)
    print("Temiz satır:", len(df))
    print("Tarih aralığı:", df["date"].min(), "->", df["date"].max())

    print("=== 3) GRID ===")
    df, x0, y0 = add_utm_and_grid(df)
    print("Grid hücre sayısı:", df["cell_id"].nunique())

    print("=== 4) WEEKLY ===")
    df = add_week_bucket(df)
    weekly = make_weekly_counts(df)
    print("Weekly satır:", len(weekly))
    print("Weekly week aralığı:", weekly["week_start"].min(), "->", weekly["week_start"].max())
    print("Weekly crime_count sum:", int(weekly["crime_count"].sum()))
    print("Weekly week_start head:", weekly["week_start"].head().tolist())

    # ✅ Aktif hücre seçimini weekly'den yap
    keep_cells = select_active_cells_from_weekly(weekly, MIN_TOTAL_CRIMES_PER_CELL)
    if len(keep_cells) == 0:
        print("WARNING: 0 cell kept. Lowering threshold to 5 for debug.")
        keep_cells = select_active_cells_from_weekly(weekly, 5)

    weekly = weekly[weekly["cell_id"].isin(keep_cells)].copy()

    print("=== 5) PANEL ===")
    panel = make_full_panel(weekly)
    print("Panel boyutu:", panel.shape)
    print("PANEL sum:", int(panel["crime_count"].sum()), "nonzero:", int((panel["crime_count"] > 0).sum()))
    print("Panel week_start head:", panel["week_start"].head().tolist())

    print("=== 6) FEATURES ===")
    panel = add_time_features(panel)
    panel = add_lag_roll_features(panel)
    panel = add_neighbor_mean(panel)

    panel.to_parquet(OUT_PANEL_PATH, index=False)
    print(f"✅ Saved: {OUT_PANEL_PATH}")

    train, val, test = time_split(panel)

    feature_cols = [
        "month","weekofyear",
        "lag_1","lag_2","lag_4","lag_8","lag_12",
        "roll_mean_4","roll_mean_12","roll_mean_24","roll_std_12",
        "neighbor_mean",
    ]

    print("=== 7) TRAIN ===")
    model = train_model(train, val, feature_cols)

    print("=== 8) EVAL ===")
    pred = evaluate(model, test, feature_cols)

    print("=== 9) SAVE PRED ===")
    save_predictions_geo(test, pred, x0, y0)

    print("\nBİTTİ ✅ Regression artık gerçek metrik üretiyor.")


if __name__ == "__main__":
    main()