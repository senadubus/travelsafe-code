import pandas as pd

# Dosya yolunu kontrol et
path = "predictions_with_geo.parquet"

df = pd.read_parquet(path)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nHead:")
print(df.head())

print("\nDescribe (sayısal kolonlar):")
print(df.describe())


df = pd.read_parquet("predictions_with_geo.parquet")

top20 = df.sort_values("pred_count", ascending=False).head(20)

print(top20[["week_start", "cell_id", "pred_count", "lat_center", "lon_center"]])