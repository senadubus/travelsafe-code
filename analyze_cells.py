import pandas as pd
import folium

PARQUET_PATH = "predictions_with_geo.parquet"
OUT_HTML = "risk_map.html"

TOP_N = 50          # en riskli kaç hücreyi göstereyim?
MIN_PRED = None     # istersen eşik: örn 5.0 (None = kapalı)

df = pd.read_parquet(PARQUET_PATH)

# Son haftayı seç
last_week = df["week_start"].max()
d = df[df["week_start"] == last_week].copy()

# Opsiyonel eşik
if MIN_PRED is not None:
    d = d[d["pred_count"] >= MIN_PRED]

# Top N
d = d.sort_values("pred_count", ascending=False).head(TOP_N)

print("Week:", last_week)
print("Showing rows:", len(d))
print(d[["cell_id", "pred_count", "lat_center", "lon_center"]].head(10))

# Harita merkezi (seçilen noktaların ortalaması)
center_lat = float(d["lat_center"].mean())
center_lon = float(d["lon_center"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Noktaları ekle
for _, row in d.iterrows():
    lat = float(row["lat_center"])
    lon = float(row["lon_center"])
    pred = float(row["pred_count"])
    cell = str(row["cell_id"])

    # pred büyüdükçe marker büyüsün diye radius ayarı (çok abartmadan)
    radius = max(4, min(18, pred / 2.0))

    popup = f"""
    <b>Cell:</b> {cell}<br>
    <b>Week:</b> {last_week.date()}<br>
    <b>Predicted count:</b> {pred:.2f}
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        popup=folium.Popup(popup, max_width=300),
        tooltip=f"{cell} | {pred:.2f}",
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

m.save(OUT_HTML)
print(f"✅ Saved map: {OUT_HTML}")