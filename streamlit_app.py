import streamlit as st
import folium
from folium.plugins import Draw, Fullscreen, MiniMap
from streamlit_folium import st_folium
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import rasterio.transform
from shapely.geometry import Polygon, box
import numpy as np
import matplotlib.cm as cm
import io
import base64
import os
from pyproj import Transformer
from PIL import Image

# ---------- Page config ----------
st.set_page_config(
    page_title="AquaScan",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0b0f1a;
    color: #c8d6e5;
}

section[data-testid="stSidebar"] {
    background: #0d1321;
    border-right: 1px solid #1e2d40;
}

section[data-testid="stSidebar"] * {
    color: #8fa8c0 !important;
}

.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4a7fa5;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-unit {
    font-size: 0.75rem;
    color: #4a7fa5;
    margin-left: 4px;
}

.status-banner {
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 8px 0;
    font-family: 'DM Mono', monospace;
}

.status-success {
    background: #0d2b1f;
    border: 1px solid #166534;
    color: #4ade80;
}

.status-error {
    background: #2b0d0d;
    border: 1px solid #991b1b;
    color: #f87171;
}

.status-info {
    background: #0d1f2b;
    border: 1px solid #1e3a5f;
    color: #7dd3fc;
}

.legend-bar {
    height: 12px;
    border-radius: 4px;
    background: linear-gradient(to right, #440154, #31688e, #35b779, #fde725);
    margin: 6px 0;
}

.legend-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #4a7fa5;
    font-family: 'DM Mono', monospace;
}

.aquascan-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #f0f9ff;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0;
}

.aquascan-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 24px;
}

div[data-testid="stButton"] > button {
    background: #0ea5e9;
    color: #000;
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    border-radius: 6px;
    width: 100%;
    transition: all 0.2s;
}

div[data-testid="stButton"] > button:hover {
    background: #38bdf8;
    transform: translateY(-1px);
}

.stFileUploader {
    border: 1px dashed #1e3a5f !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------- Helper functions ----------

def transform_coords(coords_4326, to_crs):
    transformer = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)
    return [list(transformer.transform(lng, lat)) for lng, lat in coords_4326]


def get_raster_info(raster_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        nodata = src.nodata
        width = src.width
        height = src.height
        # Also get bounds reprojected to 4326 for map centering
        bounds_4326 = transform_bounds(crs, "EPSG:4326", *bounds)
    return bounds, crs, nodata, width, height, bounds_4326


def array_to_image_overlay(img_2d, cmap_name="viridis"):
    """Convert a 2D numpy/masked array to a base64 PNG with transparency."""
    if isinstance(img_2d, np.ma.MaskedArray):
        data = img_2d.filled(np.nan)
        mask_arr = img_2d.mask
    else:
        data = img_2d.astype(float)
        mask_arr = np.isnan(data)

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if vmax == vmin:
        vmax = vmin + 1e-9

    normalised = (data - vmin) / (vmax - vmin)
    colormap = cm.get_cmap(cmap_name)
    rgba = colormap(normalised)  # (H, W, 4)

    # Transparent where masked/nodata
    if mask_arr is not False and mask_arr.any():
        rgba[mask_arr, 3] = 0.0
    else:
        rgba[np.isnan(data), 3] = 0.0

    rgba_uint8 = (rgba * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgba_uint8, mode="RGBA")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}", vmin, vmax


def clip_raster(raster_path, polygon_coords_4326):
    """Clip raster to polygon and return image data, stats, overlay bounds, and transform."""
    if not polygon_coords_4326:
        return None, "No polygon drawn"

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if raster_crs is None:
            return None, "Raster has no CRS defined"

        raster_crs_str = raster_crs.to_string() if hasattr(raster_crs, "to_string") else str(raster_crs)

        try:
            transformed = transform_coords(polygon_coords_4326, raster_crs_str)
            poly = Polygon(transformed)
        except Exception as e:
            return None, f"Coordinate transformation failed: {e}"

        left, bottom, right, top = src.bounds
        raster_box = box(left, bottom, right, top)

        if not poly.intersects(raster_box):
            return None, (
                f"Polygon is outside the raster extent.\n\n"
                f"Raster bounds ({raster_crs_str}): "
                f"({left:.4f}, {bottom:.4f}, {right:.4f}, {top:.4f})\n"
                f"Polygon bounds: {tuple(round(v, 4) for v in poly.bounds)}"
            )

        try:
            out_image, out_transform = mask(src, [poly], crop=True, all_touched=True)
        except Exception as e:
            return None, f"Mask operation failed: {e}"

        nodata = src.nodata
        if nodata is not None:
            out_image = np.ma.masked_equal(out_image, nodata)
        else:
            out_image = np.ma.masked_invalid(out_image.astype(float))

        if out_image.size == 0 or out_image.mask.all():
            return None, "Clipping returned no valid data — all pixels are masked or nodata."

        img_2d = out_image[0]

        # Stats
        mean_val = float(img_2d.mean())
        min_val = float(img_2d.min())
        max_val = float(img_2d.max())
        std_val = float(img_2d.std())

        # Compute overlay bounds in EPSG:4326
        h, w = img_2d.shape
        raw_bounds = rasterio.transform.array_bounds(h, w, out_transform)
        bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *raw_bounds)

        return {
            "img_2d": img_2d,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "bounds_4326": bounds_4326,  # (west, south, east, north)
        }, None


def build_map(center_lat, center_lng, zoom, overlay_data=None, polygon_coords=None):
    """Build the Folium map, optionally with soil moisture overlay."""
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles=None
    )

    # Base tiles
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Dark",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    # Draw control
    Draw(
        draw_options={
            "polygon": {"allowIntersection": False, "shapeOptions": {"color": "#38bdf8", "weight": 2}},
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True},
    ).add_to(m)

    Fullscreen().add_to(m)

    # Soil moisture overlay
    if overlay_data:
        img_b64, vmin, vmax = overlay_data["img_b64"], overlay_data["vmin"], overlay_data["vmax"]
        west, south, east, north = overlay_data["bounds_4326"]

        folium.raster_layers.ImageOverlay(
            image=img_b64,
            bounds=[[south, west], [north, east]],
            opacity=0.75,
            name="Soil Moisture",
            interactive=True,
            cross_origin=False,
            zindex=10,
        ).add_to(m)

    # Re-draw the polygon if we have one
    if polygon_coords:
        folium.Polygon(
            locations=[[lat, lng] for lng, lat in polygon_coords],
            color="#38bdf8",
            weight=2,
            fill=True,
            fill_opacity=0.05,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ---------- State defaults ----------
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "polygon_coords" not in st.session_state:
    st.session_state.polygon_coords = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [53.3498, -6.2603]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 8


# ---------- Header ----------
st.markdown('<div class="aquascan-header">💧 AquaScan</div>', unsafe_allow_html=True)
st.markdown('<div class="aquascan-sub">Soil Moisture Analysis Platform</div>', unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📁 Data Source")

    # Raster loading: bundled file takes priority, fallback to uploader
    BUNDLED_RASTER = "soil_moisture.tif"
    raster_path = None

    if os.path.exists(BUNDLED_RASTER):
        raster_path = BUNDLED_RASTER
        st.markdown(
            '<div class="status-banner status-success">✓ soil_moisture.tif loaded</div>',
            unsafe_allow_html=True
        )
    else:
        uploaded_file = st.file_uploader("Upload GeoTIFF (.tif)", type=["tif", "tiff"])
        if uploaded_file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(uploaded_file.getvalue())
                raster_path = tmp.name
            st.markdown(
                '<div class="status-banner status-success">✓ Raster loaded from upload</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-banner status-info">Place soil_moisture.tif in the app directory, or upload a GeoTIFF above.</div>',
                unsafe_allow_html=True
            )

    # Raster metadata
    if raster_path:
        try:
            bounds, crs, nodata, w, h, bounds_4326 = get_raster_info(raster_path)
            west, south, east, north = bounds_4326
            center_lat = (south + north) / 2
            center_lng = (west + east) / 2
            st.session_state.map_center = [center_lat, center_lng]

            with st.expander("Raster metadata", expanded=False):
                st.code(
                    f"CRS:      {crs}\n"
                    f"Size:     {w} × {h} px\n"
                    f"Nodata:   {nodata}\n"
                    f"Bounds:   {west:.4f}W  {east:.4f}E\n"
                    f"          {south:.4f}S  {north:.4f}N",
                    language=None
                )
        except Exception as e:
            st.markdown(
                f'<div class="status-banner status-error">Cannot read raster: {e}</div>',
                unsafe_allow_html=True
            )
            raster_path = None

    st.markdown("---")
    st.markdown("### 🗺️ Instructions")
    st.markdown("""
<div style="font-size:0.78rem; color:#6b8fa8; line-height:1.7;">
1. Draw a polygon on the map using the toolbar (top-left)<br>
2. Click <strong style="color:#38bdf8;">Analyse Area</strong> below<br>
3. The soil moisture layer will appear overlaid on the map
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    opacity_val = st.slider("Overlay opacity", 0.1, 1.0, 0.75, 0.05)

    run_button = st.button("▶ Analyse Area", type="primary", disabled=(raster_path is None))

    st.markdown("---")
    st.markdown("### 🎨 Legend")
    st.markdown("""
<div class="legend-bar"></div>
<div class="legend-labels">
  <span>Dry</span><span>Moderate</span><span>Wet</span>
</div>
<div style="font-size:0.65rem; color:#4a7fa5; margin-top:4px;">Viridis scale · volumetric water content</div>
""", unsafe_allow_html=True)


# ---------- Map area ----------
col_map, col_stats = st.columns([3, 1])

with col_map:
    overlay_data = None
    if st.session_state.result:
        res = st.session_state.result
        img_b64, vmin, vmax = array_to_image_overlay(res["img_2d"], cmap_name="viridis")
        overlay_data = {
            "img_b64": img_b64,
            "vmin": vmin,
            "vmax": vmax,
            "bounds_4326": res["bounds_4326"],
        }

    m = build_map(
        center_lat=st.session_state.map_center[0],
        center_lng=st.session_state.map_center[1],
        zoom=st.session_state.map_zoom,
        overlay_data=overlay_data,
        polygon_coords=st.session_state.polygon_coords,
    )

    map_output = st_folium(m, width="100%", height=560, returned_objects=["last_active_drawing"])

    # Capture drawn polygon
    if map_output and map_output.get("last_active_drawing"):
        geom = map_output["last_active_drawing"]["geometry"]
        if geom["type"] == "Polygon":
            st.session_state.polygon_coords = geom["coordinates"][0]


# ---------- Run analysis ----------
if run_button:
    if raster_path is None:
        st.session_state.error = "No raster file available."
        st.session_state.result = None
    elif not st.session_state.polygon_coords:
        st.session_state.error = "Draw a polygon on the map first."
        st.session_state.result = None
    else:
        with st.spinner("Clipping raster to polygon..."):
            result, error = clip_raster(raster_path, st.session_state.polygon_coords)
        if error:
            st.session_state.error = error
            st.session_state.result = None
        else:
            st.session_state.result = result
            st.session_state.error = None
        st.rerun()


# ---------- Stats panel ----------
with col_stats:
    st.markdown("#### Analysis Results")

    if st.session_state.error:
        st.markdown(
            f'<div class="status-banner status-error">⚠ {st.session_state.error}</div>',
            unsafe_allow_html=True
        )

    if st.session_state.result:
        res = st.session_state.result

        metrics = [
            ("Mean moisture", res["mean"], "m³/m³"),
            ("Min", res["min"], "m³/m³"),
            ("Max", res["max"], "m³/m³"),
            ("Std dev", res["std"], ""),
        ]

        for label, value, unit in metrics:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value:.4f}<span class="metric-unit">{unit}</span></div>
</div>
""", unsafe_allow_html=True)

        # Moisture interpretation
        mean = res["mean"]
        if mean < 0.15:
            interpretation = "🟡 Dry — consider irrigation"
            interp_class = "status-error"
        elif mean < 0.30:
            interpretation = "🟢 Adequate moisture"
            interp_class = "status-success"
        else:
            interpretation = "🔵 High moisture / saturation risk"
            interp_class = "status-info"

        st.markdown(f"""
<div class="status-banner {interp_class}" style="margin-top:16px;">
  {interpretation}
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="color:#2d4a63; font-size:0.78rem; font-family:'DM Mono',monospace; 
     padding: 24px 0; text-align:center; line-height:2;">
  Draw a polygon<br>then click<br>Analyse Area
</div>
""", unsafe_allow_html=True)
