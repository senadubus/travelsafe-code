#!/usr/bin/env python3
"""
Chicago yol ağını OpenStreetMap'ten çekip PostgreSQL/PostGIS'e yazar.

Ne yapar?
1) OSMnx ile Chicago driving network indirir
2) Graph'i nodes ve edges GeoDataFrame'lerine çevirir
3) Kolonları temizler / isimleri sadeleştirir
4) PostGIS'te road_nodes ve road_segments tablolarına yazar
5) GIST index ve bazı yardımcı kolonları oluşturur

Gereksinimler:
- osmnx
- geopandas
- sqlalchemy
- geoalchemy2
- psycopg2
- PostgreSQL + PostGIS

Çalıştırma:
python import_chicago_roads.py
"""

from __future__ import annotations

import sys
import traceback
from typing import Iterable

import geopandas as gpd
import osmnx as ox
import pandas as pd
from sqlalchemy import create_engine, text


# =========================
# AYARLAR
# =========================
DB_USER = "postgres"
DB_PASSWORD = "yeni_sifre"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "travelsafe"

DB_SCHEMA = "public"

PLACE_NAME = "Chicago, Illinois, USA"
NETWORK_TYPE = "all"   # drive / walk / bike / all
SIMPLIFY = True

NODES_TABLE = "road_nodes"
EDGES_TABLE = "road_segments"


# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def make_engine():
    """
    SQLAlchemy engine oluşturur.
    """
    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine


def ensure_postgis(engine) -> None:
    """
    PostGIS eklentisini açar.
    """
    sql = "CREATE EXTENSION IF NOT EXISTS postgis;"
    with engine.begin() as conn:
        conn.execute(text(sql))


def flatten_listlike(value):
    """
    OSM verisindeki bazı kolonlar list gelebilir.
    Bunları string'e çeviriyoruz ki Postgres'e yazması kolay olsun.
    """
    if isinstance(value, (list, tuple, set)):
        return ",".join(map(str, value))
    return value


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Object kolonlardaki list/tuple/set değerleri düzleştirir.
    """
    df = df.copy()
    for col in df.columns:
        if col == "geometry":
            continue
        if df[col].dtype == "object":
            df[col] = df[col].map(flatten_listlike)
    return df


def normalize_nodes_gdf(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Nodes GeoDataFrame kolonlarını sadeleştirir.
    OSMnx graph_to_gdfs sonrası node index'i genellikle osmid'dir.
    """
    nodes = nodes.copy()

    # index'i kolona al
    nodes = nodes.reset_index()

    # osmid kolonunu tek isimde tut
    if "osmid" not in nodes.columns:
        # bazen index adı farklı olabilir
        first_col = nodes.columns[0]
        nodes = nodes.rename(columns={first_col: "osmid"})

    # geometri ve temel alanlar dışında fazla karmaşık kolonlar kalabilir
    nodes = clean_object_columns(nodes)

    # Kolon isimlerini daha SQL-dostu hale getir
    rename_map = {}
    for c in nodes.columns:
        new_c = (
            c.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        rename_map[c] = new_c
    nodes = nodes.rename(columns=rename_map)

    # yardımcı x/y kolonları ekle
    if "x" not in nodes.columns:
        nodes["x"] = nodes.geometry.x
    if "y" not in nodes.columns:
        nodes["y"] = nodes.geometry.y

    return nodes


def normalize_edges_gdf(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Edges GeoDataFrame kolonlarını sadeleştirir.
    OSMnx edge tarafında genelde index: (u, v, key)
    """
    edges = edges.copy()

    # MultiIndex'i kolonlara al
    edges = edges.reset_index()

    edges = clean_object_columns(edges)

    # Kolon adlarını sadeleştir
    rename_map = {}
    for c in edges.columns:
        new_c = (
            c.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        rename_map[c] = new_c
    edges = edges.rename(columns=rename_map)

    # Beklediğimiz kolonları garanti altına al
    expected = ["u", "v", "key", "length", "geometry"]
    for col in expected:
        if col not in edges.columns:
            print(f"Uyarı: '{col}' kolonu bulunamadı.")

    # A* ve ileride pgRouting için yardımcı kolonlar
    if "source_node" not in edges.columns and "u" in edges.columns:
        edges["source_node"] = edges["u"]

    if "target_node" not in edges.columns and "v" in edges.columns:
        edges["target_node"] = edges["v"]

    # metre cinsinden uzunluk OSMnx edge'lerinde çoğunlukla 'length' olarak gelir
    if "length_m" not in edges.columns:
        if "length" in edges.columns:
            edges["length_m"] = edges["length"]
        else:
            edges["length_m"] = None

    # başlangıç maliyetleri
    if "base_cost" not in edges.columns:
        edges["base_cost"] = edges["length_m"]

    if "safety_cost" not in edges.columns:
        edges["safety_cost"] = 0.0

    if "total_cost" not in edges.columns:
        # ilk aşamada mesafe bazlı
        edges["total_cost"] = edges["length_m"]

    return edges


def write_gdf_to_postgis(
    gdf: gpd.GeoDataFrame,
    table_name: str,
    engine,
    schema: str = "public",
    if_exists: str = "replace",
    chunksize: int = 5000,
) -> None:
    """
    GeoDataFrame'i PostGIS'e yazar.
    """
    gdf.to_postgis(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
    )


def add_post_import_sql(engine, schema: str, nodes_table: str, edges_table: str) -> None:
    """
    Tablolara primary key, index ve yardımcı kolonlar ekler.
    """
    sql_statements: Iterable[str] = [
        # -------------------------
        # NODE TABLOSU
        # -------------------------
        f"""
        ALTER TABLE {schema}.{nodes_table}
        ADD COLUMN IF NOT EXISTS id BIGSERIAL;
        """,
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = '{nodes_table}_pkey'
            ) THEN
                ALTER TABLE {schema}.{nodes_table}
                ADD CONSTRAINT {nodes_table}_pkey PRIMARY KEY (id);
            END IF;
        END $$;
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{nodes_table}_geom
        ON {schema}.{nodes_table}
        USING GIST (geometry);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{nodes_table}_osmid
        ON {schema}.{nodes_table} (osmid);
        """,

        # -------------------------
        # EDGE TABLOSU
        # -------------------------
        f"""
        ALTER TABLE {schema}.{edges_table}
        ADD COLUMN IF NOT EXISTS id BIGSERIAL;
        """,
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = '{edges_table}_pkey'
            ) THEN
                ALTER TABLE {schema}.{edges_table}
                ADD CONSTRAINT {edges_table}_pkey PRIMARY KEY (id);
            END IF;
        END $$;
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{edges_table}_geom
        ON {schema}.{edges_table}
        USING GIST (geometry);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{edges_table}_source
        ON {schema}.{edges_table} (source_node);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{edges_table}_target
        ON {schema}.{edges_table} (target_node);
        """,
    ]

    with engine.begin() as conn:
        for stmt in sql_statements:
            conn.execute(text(stmt))


def print_summary(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> None:
    """
    Basit özet basar.
    """
    print("\n=== ÖZET ===")
    print(f"Node sayısı : {len(nodes):,}")
    print(f"Edge sayısı : {len(edges):,}")
    print(f"Node CRS    : {nodes.crs}")
    print(f"Edge CRS    : {edges.crs}")

    print("\nNode kolonları:")
    print(list(nodes.columns))

    print("\nEdge kolonları:")
    print(list(edges.columns))


# =========================
# ANA AKIŞ
# =========================
def main():
    try:
        print("1) SQLAlchemy engine oluşturuluyor...")
        engine = make_engine()

        print("2) PostGIS kontrol ediliyor...")
        ensure_postgis(engine)

        print(f"3) OSMnx ile yol ağı çekiliyor: {PLACE_NAME} | network_type={NETWORK_TYPE}")
        # OSMnx resmi user reference'ta graph_from_place destekliyor
        G = ox.graph_from_place(
            PLACE_NAME,
            network_type=NETWORK_TYPE,
            simplify=SIMPLIFY,
        )

        print("4) Graph -> GeoDataFrame dönüşümü yapılıyor...")
        nodes, edges = ox.graph_to_gdfs(G)

        print("5) GeoDataFrame'ler normalize ediliyor...")
        nodes = normalize_nodes_gdf(nodes)
        edges = normalize_edges_gdf(edges)

        print_summary(nodes, edges)

        print(f"\n6) {DB_SCHEMA}.{NODES_TABLE} tablosuna node'lar yazılıyor...")
        write_gdf_to_postgis(
            gdf=nodes,
            table_name=NODES_TABLE,
            engine=engine,
            schema=DB_SCHEMA,
            if_exists="replace",
            chunksize=5000,
        )

        print(f"7) {DB_SCHEMA}.{EDGES_TABLE} tablosuna edge'ler yazılıyor...")
        write_gdf_to_postgis(
            gdf=edges,
            table_name=EDGES_TABLE,
            engine=engine,
            schema=DB_SCHEMA,
            if_exists="replace",
            chunksize=5000,
        )

        print("8) Post-import index ve yardımcı SQL çalıştırılıyor...")
        add_post_import_sql(
            engine=engine,
            schema=DB_SCHEMA,
            nodes_table=NODES_TABLE,
            edges_table=EDGES_TABLE,
        )

        print("\nTamamlandı ✅")
        print(f"Tablolar:")
        print(f"- {DB_SCHEMA}.{NODES_TABLE}")
        print(f"- {DB_SCHEMA}.{EDGES_TABLE}")

        print("\nSonraki adım:")
        print("road_segments ile grid_cells arasında kesişim tablosu üret.")
        print("Sonra safety_cost ve total_cost alanlarını güncelle.")

    except Exception as e:
        print("\nHATA OLDU ❌")
        print(str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()