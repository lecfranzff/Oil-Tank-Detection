"""Small geography helpers: GSD, pixel <-> lon/lat, distances, DMS parsing."""
import math
import re

R_EARTH = 6378137.0   # WGS84 equatorial radius, metres (same constant web-mercator uses)


def true_gsd(merc_res, lat):
    """Web-mercator pixels are stretched by 1/cos(lat); ground size = merc res x cos(lat)."""
    return merc_res * math.cos(math.radians(lat))


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lon/lat points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def parse_dms(s):
    """'29°22'16.84"N' -> 29.371344. South and West are negative."""
    m = re.match(r"\s*(\d+)\D+(\d+)\D+([\d.]+)\D+([NSEW])", s)
    d, mi, se, h = m.groups()
    v = int(d) + int(mi) / 60 + float(se) / 3600
    return -v if h in "SW" else v


def lonlat_to_merc(lon, lat):
    """EPSG:4326 -> EPSG:3857 metres."""
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def merc_to_lonlat(x, y):
    """EPSG:3857 metres -> EPSG:4326."""
    lon = math.degrees(x / R_EARTH)
    lat = math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2)
    return lon, lat


def pixel_to_lonlat(transform, col, row):
    """Pixel (col, row) of an EPSG:3857 GeoTIFF -> (lon, lat). Uses the pixel centre."""
    x, y = transform @ (col + 0.5, row + 0.5)
    return merc_to_lonlat(x, y)


def resample_geotiff(src_path, dst_path, target_gsd, lat):
    """Resample an EPSG:3857 GeoTIFF so one pixel covers `target_gsd` metres on the ground.

    In web-mercator the stored pixel size is the *mercator* size, so the stored res we
    want is target_gsd / cos(lat). Downsampling uses area averaging. Returns the new
    stored (mercator) resolution.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(src_path) as src:
        merc_res = target_gsd / math.cos(math.radians(lat))
        scale = src.res[0] / merc_res                      # < 1 means downsampling
        width, height = round(src.width * scale), round(src.height * scale)
        data = src.read(out_shape=(src.count, height, width), resampling=Resampling.average)
        # stretch the pixel size so the image still covers exactly the same ground
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        profile = src.profile
        profile.update(width=width, height=height, transform=transform)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)
    return transform.a


def lonlat_to_pixel(transform, lon, lat):
    """Inverse of pixel_to_lonlat: (lon, lat) -> fractional (col, row)."""
    x, y = lonlat_to_merc(lon, lat)
    col, row = ~transform @ (x, y)
    return col - 0.5, row - 0.5
