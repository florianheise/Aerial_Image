#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
import unicodedata
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image
import requests

import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling


# ------------------------------------------------------
# TQDM FALLBACK
# ------------------------------------------------------
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        if iterable is None:
            class Dummy:
                def __enter__(self): return self
                def __exit__(self, t, v, tb): return False
                def update(self, n=1): pass
            return Dummy()
        return iterable


# ------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------
TILE_URL = "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"  # note: real '&'
TILE_SIZE = 256

# Your target zone (for Brisbane/SE QLD this is 56).
TARGET_ZONE = 56

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "imagery-tool/accurate-6.0"})


# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def sanitize_project_no(p: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "", p.strip())


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_outputs(project_no: str):
    ts = now_timestamp()
    p = sanitize_project_no(project_no)
    return {
        "png":  f"{p}_preview_{ts}.png",
        "tif":  f"{p}_final_{ts}.tif",
        "meta": f"{p}_metadata_{ts}.txt",
    }


def dms_to_decimal(d: str) -> float:
    """
    Parse DMS or decimal with optional hemisphere.
    Examples: "27°50'49.38\"S", "153°17'57.38\"E", "-27.85", "153.30E"
    """
    s = unicodedata.normalize("NFKD", d.strip())
    s = (s.replace("″", "\"").replace("”", "\"").replace("“", "\"")
           .replace("′", "'").replace("’", "'").replace("‘", "'"))
    s = re.sub(r"\s+", " ", s)

    m = re.match(
        r'^\s*(?P<deg>[+-]?\d+(?:\.\d+)?)\s*[°]?\s*'
        r'(?P<min>\d+(?:\.\d+)?)?\s*[\']?\s*'
        r'(?P<sec>\d+(?:\.\d+)?)?\s*["]?\s*'
        r'(?P<hem>[NnSsEeWw])?\s*$',
        s
    )
    if not m:
        raise ValueError(f"Invalid DMS: {d}")

    deg = float(m.group("deg"))
    mn  = float(m.group("min") or 0)
    sec = float(m.group("sec") or 0)
    hem = (m.group("hem") or "").upper()

    dec = abs(deg) + mn/60 + sec/3600
    if hem in ("S", "W") or (hem == "" and str(deg).startswith("-")):
        dec = -dec
    return dec


# ------------------------------------------------------
# TILE MATH
# ------------------------------------------------------
def latlon_to_tile(lat, lon, z):
    """Lat/Lon → XYZ tile index (Google/Web Mercator style)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2 ** z
    xt = int((lon + 180.0) / 360.0 * n)
    yt = int((1.0 - (math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)) / 2.0 * n)
    return xt, yt


def approx_bbox_m(lat, lon, metres):
    """Approximate bbox (south, north, west, east) in degrees for a metre radius."""
    dlat = metres / 111320.0
    dlon = metres / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def tile_to_latlon(x, y, z):
    """XYZ tile top-left corner → WGS84 (lat, lon)."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2*y/n)))
    lat = math.degrees(lat_rad)
    return lat, lon


# ------------------------------------------------------
# TILE DOWNLOAD
# ------------------------------------------------------
def download_tile(x, y, z):
    try:
        r = SESSION.get(TILE_URL.format(x=x, y=y, z=z), timeout=10)
        r.raise_for_status()
        if "image" not in r.headers.get("Content-Type", ""):
            return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# ------------------------------------------------------
# CRS RESOLUTION & FALLBACKS
# ------------------------------------------------------
def candidate_dst_crs_list(zone: int):
    """
    Returns a prioritized list of candidate projected CRS (metres) for the given UTM zone.
    1) GDA2020 / MGA zone N (EPSG:784x) — only adds 7844 for zone 56 (your region)
    2) WGS84 / UTM zone N South (EPSG:32700 + zone) — widely available
    3) Web Mercator (EPSG:3857) — last resort, meters
    """
    candidates = []

    # GDA2020 / MGA56 only (known code). For other zones you can add mappings if needed.
    if zone == 56:
        candidates.append(("EPSG:7844", "GDA2020 / MGA Zone 56"))

    # WGS84 / UTM (South) general fallback (EPSG:32700 + zone)
    candidates.append((f"EPSG:{32700 + zone}", f"WGS84 / UTM Zone {zone}S"))

    # Web Mercator ultimate fallback (meters)
    candidates.append(("EPSG:3857", "Web Mercator"))

    return candidates


def try_reproject_4326_to(dst_crs_str, bands, width_px, height_px,
                          west_lon, south_lat, east_lon, north_lat, transform4326, out_path):
    """
    Attempts reprojection from EPSG:4326 (with provided transform) to dst_crs_str.
    Returns (ok: bool, gsd_x, gsd_y, width_m, height_m, info_str).
    """
    try:
        dst_crs = CRS.from_string(dst_crs_str)
    except Exception as e:
        return False, None, None, None, None, f"Could not parse CRS {dst_crs_str}: {e}"

    # Compute output grid & transform
    try:
        t_dst, w_dst, h_dst = calculate_default_transform(
            CRS.from_epsg(4326),
            dst_crs,
            width_px, height_px,
            west_lon, south_lat, east_lon, north_lat
        )
    except Exception as e:
        return False, None, None, None, None, f"calculate_default_transform failed for {dst_crs_str}: {e}"

    profile_dst = dict(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=3,
        width=w_dst,
        height=h_dst,
        crs=dst_crs,
        transform=t_dst,
    )

    # Reproject
    try:
        with rasterio.open(out_path, "w", **profile_dst) as dst:
            for b in range(3):
                reproject(
                    source=bands[b],
                    destination=rasterio.band(dst, b + 1),
                    src_transform=transform4326,
                    src_crs=CRS.from_epsg(4326),
                    dst_transform=t_dst,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    except Exception as e:
        return False, None, None, None, None, f"reproject failed for {dst_crs_str}: {e}"

    # Inspect size in meters via bounds
    try:
        with rasterio.open(out_path) as gda:
            gsd_x = float(gda.transform.a)
            gsd_y = float(-gda.transform.e)
            b = gda.bounds
            width_m  = float(b.right - b.left)
            height_m = float(b.top - b.bottom)
    except Exception as e:
        return False, None, None, None, None, f"inspect bounds failed: {e}"

    # Reject unrealistic outputs (e.g., if CRS was not applied correctly)
    if width_m < 1 or height_m < 1:
        return False, gsd_x, gsd_y, width_m, height_m, (
            f"Projected dimensions too small "
            f"(w={width_m:.6f} m, h={height_m:.6f} m) for {dst_crs_str}"
        )

    return True, gsd_x, gsd_y, width_m, height_m, f"OK {dst_crs_str}"


# ------------------------------------------------------
# METADATA
# ------------------------------------------------------
def write_metadata(path, gsd_m_per_px, w_px, h_px, w_m, h_m):
    w_mm = w_m * 1000.0
    h_mm = h_m * 1000.0
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Width_px: {w_px}\n")
        f.write(f"Height_px: {h_px}\n")
        f.write(f"Width_m: {w_m:.3f}\n")
        f.write(f"Height_m: {h_m:.3f}\n")
        f.write(f"Width_mm: {w_mm:.1f}\n")
        f.write(f"Height_mm: {h_mm:.1f}\n")
        f.write(f"GSD (m/pixel): {gsd_m_per_px:.6f}\n")


# ------------------------------------------------------
# EXPECTED SIZE (pre-check using pure Mercator math; no PROJ)
# ------------------------------------------------------
def print_expected_size_mercator(tiles_x, tiles_y, zoom):
    """Compute expected ground width/height in meters using Web Mercator resolution."""
    res_m_per_px = 156543.03392804097 / (2 ** zoom)  # meters per pixel
    tile_w_m = TILE_SIZE * res_m_per_px

    expected_w_m = tiles_x * tile_w_m
    expected_h_m = tiles_y * tile_w_m

    print("\nEXPECTED (pre-download) real-world size [Mercator math]:")
    print(f"  Width:  {expected_w_m:.3f} m")
    print(f"  Height: {expected_h_m:.3f} m")
    print(f"  Approx GSD: {res_m_per_px:.4f} m/px at zoom {zoom}\n")


# ------------------------------------------------------
# MAIN WORKFLOW — Option B (final projected GeoTIFF only)
# ------------------------------------------------------
def download_stitch_and_export(lat, lon, extent_m, zoom, outputs):
    # 1) Select tiles covering the extent
    south, north, west, east = approx_bbox_m(lat, lon, extent_m)
    tx1, ty1 = latlon_to_tile(north, west, zoom)
    tx2, ty2 = latlon_to_tile(south, east, zoom)

    xmin, xmax = sorted((tx1, tx2))
    ymin, ymax = sorted((ty1, ty2))

    tiles_x = xmax - xmin + 1
    tiles_y = ymax - ymin + 1
    width_px  = tiles_x * TILE_SIZE
    height_px = tiles_y * TILE_SIZE

    print(f"Tiles X: {xmin}..{xmax}  ({tiles_x})")
    print(f"Tiles Y: {ymin}..{ymax}  ({tiles_y})")
    print(f"Stitched image (px): {width_px} × {height_px}")

    # Pre-download sanity check: should be hundreds of meters at z=18–20
    print_expected_size_mercator(tiles_x, tiles_y, zoom)

    if input("Continue? (y/n): ").strip().lower() != "y":
        return

    # 2) Stitch tiles into an RGB mosaic
    mosaic = Image.new("RGB", (width_px, height_px), (200, 200, 200))
    got_any = False
    with tqdm(total=tiles_x * tiles_y, desc="Tiles", unit="tile") as pbar:
        for xi, X in enumerate(range(xmin, xmax + 1)):
            for yi, Y in enumerate(range(ymin, ymax + 1)):
                tile = download_tile(X, Y, zoom)
                if tile is not None:
                    mosaic.paste(tile, (xi * TILE_SIZE, yi * TILE_SIZE))
                    got_any = True
                pbar.update(1)

    if not got_any:
        raise RuntimeError("No tiles were downloaded. Check network access or TILE_URL.")

    mosaic.save(outputs["png"])
    print(f"Saved PNG preview: {outputs['png']}")

    # 3) True WGS84 geographic bounds of the stitched image (outer edges)
    north_lat, west_lon = tile_to_latlon(xmin,     ymin,     zoom)
    south_lat, east_lon = tile_to_latlon(xmax + 1, ymax + 1, zoom)

    # 4) Build correct north-up WGS84 (EPSG:4326) transform for the mosaic
    pixel_width_deg  = (east_lon - west_lon) / width_px
    pixel_height_deg = (south_lat - north_lat) / height_px  # MUST be negative
    transform4326 = Affine(
        pixel_width_deg, 0.0, west_lon,
        0.0, pixel_height_deg, north_lat
    )

    arr = np.asarray(mosaic, dtype=np.uint8)  # H x W x 3
    bands = np.transpose(arr, (2, 0, 1))      # 3 x H x W

    # 5) Decide UTM zone by longitude mid-point (for generality)
    mid_lon = (west_lon + east_lon) / 2.0
    zone = int((mid_lon + 180.0) // 6) + 1
    # keep your target (56) if you prefer forcing it:
    if TARGET_ZONE:
        zone = TARGET_ZONE

    # 6) Try CRS candidates in order (7844 → 32756 → 3857)
    candidates = candidate_dst_crs_list(zone)

    last_info = ""
    for crs_str, label in candidates:
        print(f"Trying destination CRS: {crs_str} ({label}) ...")
        ok, gsd_x, gsd_y, width_m, height_m, info = try_reproject_4326_to(
            crs_str, bands, width_px, height_px,
            west_lon, south_lat, east_lon, north_lat,
            transform4326, outputs["tif"]
        )
        print("  ", info)
        last_info = info
        if ok:
            # Success; write metadata & finish
            with rasterio.open(outputs["tif"]) as ds:
                gsd_m_per_px = max(abs(gsd_x), abs(gsd_y))
                w_px_final, h_px_final = ds.width, ds.height
            width_mm, height_mm = width_m * 1000.0, height_m * 1000.0
            print("\nFinal real-world size (from bounds):")
            print(f"  {width_m:.3f} m × {height_m:.3f} m")
            print(f"  {width_mm:.1f} mm × {height_mm:.1f} mm")
            print(f"  GSD_x: {gsd_x:.4f} m/px, GSD_y: {gsd_y:.4f} m/px (GSD: {gsd_m_per_px:.4f})\n")

            write_metadata(outputs["meta"], gsd_m_per_px, w_px_final, h_px_final, width_m, height_m)
            print(f"Saved final GeoTIFF: {outputs['tif']}")
            print(f"Metadata saved: {outputs['meta']}")
            return

    # If we ended the loop without returning, all candidates failed
    raise RuntimeError(
        "All destination CRS attempts failed to yield realistic meter-sized extents.\n"
        f"Last info: {last_info}\n"
        "Your GDAL/PROJ likely lacks GDA2020 and UTM definitions. "
        "Please update GDAL/PROJ, or install proj-datumgrid / EPSG support."
    )


# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    print("\nProject number (e.g. 0909):")
    project_no = input("> ").strip()

    print("\nLatitude in DMS (e.g. 27°50'49.38\"S) or decimal:")
    lat_str = input("> ")

    print("Longitude in DMS (e.g. 153°17'57.38\"E) or decimal:")
    lon_str = input("> ")

    zoom = int(input("Zoom (18–20): "))
    extent = float(input("Extent (meters): "))

    lat = dms_to_decimal(lat_str)
    lon = dms_to_decimal(lon_str)

    outputs = build_outputs(project_no)

    print(f"\nDecimal degrees: Lat={lat:.8f}, Lon={lon:.8f}")
    print("Output files will be:")
    print(f"  PNG:  {outputs['png']}")
    print(f"  TIF:  {outputs['tif']}")
    print(f"  META: {outputs['meta']}")

    download_stitch_and_export(lat, lon, extent, zoom, outputs)
