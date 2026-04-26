import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, box
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import tempfile
import os
from pyproj import Transformer

st.set_page_config(page_title="AquaScan – Local Soil Moisture", layout="wide")
st.title("🌧️ AquaHub - Local Soil Moisture Viewer")
st.markdown("Draw a polygon on the map to view soil moisture from your local GeoTIFF.")

# ---------- Helper functions ----------
def transform_coords(coords, from_crs, to_crs):
    """Transform a list of [x,y] pairs."""
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return [list(transformer.transform(x, y)) for x, y in coords]

def get_raster_footprint_4326(raster_path):
    """Return raster bounds as a polygon in EPSG:4326 (lat/lon) for map display."""
    with rasterio.open(raster_path) as src:
        bounds = src.bounds  # left, bottom, right, top in raster CRS
        raster_crs = src.crs
        if raster_crs is None:
            st.error("Raster has no CRS defined. Cannot determine footprint.")
            return None
        # Create polygon in raster CRS
        poly_raster = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        # Transform to WGS84 for map
        transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        # Get corner coordinates
        corners = [
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top),
            (bounds.left, bounds.bottom)
        ]
        corners_4326 = [transformer.transform(x, y) for x, y in corners]
        return Polygon(corners_4326)

def clip_raster_with_polygon(raster_path, polygon_coords_4326):
    """
    Clip the raster with a polygon in EPSG:4326.
    Returns clipped image (2D), mean, min, max.
    """
    if not polygon_coords_4326 or len(polygon_coords_4326) < 3:
        return None, None, None, None

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if raster_crs is None:
            st.error("Raster has no CRS defined. Cannot proceed.")
            return None, None, None, None
        raster_crs_str = raster_crs.to_string() if hasattr(raster_crs, 'to_string') else str(raster_crs)
        
        # Transform polygon to raster CRS
        transformed_coords = transform_coords(polygon_coords_4326, "EPSG:4326", raster_crs_str)
        poly = Polygon(transformed_coords)
        
        # Check overlap with raster bounds
        raster_bounds = src.bounds
        poly_bounds = poly.bounds
        if not (poly_bounds[0] < raster_bounds[2] and poly_bounds[2] > raster_bounds[0] and
                poly_bounds[1] < raster_bounds[3] and poly_bounds[3] > raster_bounds[1]):
            # No overlap – show detailed info
            st.error("❌ The drawn polygon does not overlap the raster extent.")
            st.info(f"Raster bounds (in its CRS {raster_crs_str}): left={raster_bounds[0]:.2f}, bottom={raster_bounds[1]:.2f}, right={raster_bounds[2]:.2f}, top={raster_bounds[3]:.2f}")
            st.info(f"Polygon bounds (in raster CRS): xmin={poly_bounds[0]:.2f}, ymin={poly_bounds[1]:.2f}, xmax={poly_bounds[2]:.2f}, ymax={poly_bounds[3]:.2f}")
            return None, None, None, None
        
        # Clip
        out_image, out_transform = mask(src, [poly], crop=True, all_touched=True)
        nodata = src.nodata
        if nodata is not None:
            out_image = np.ma.masked_equal(out_image, nodata)
        else:
            out_image = np.ma.masked_array(out_image, np.isnan(out_image))
        
        if out_image.size == 0 or out_image.mask.all():
            st.error("Clipping resulted in no data. Polygon may be outside raster.")
            return None, None, None, None

        mean_val = float(out_image.mean())
        min_val = float(out_image.min())
        max_val = float(out_image.max())
        img_2d = out_image[0] if out_image.shape[0] > 0 else out_image
        return img_2d, mean_val, min_val, max_val

# ---------- Raster file selection ----------
st.sidebar.header("📁 Soil Moisture Raster")
uploaded_file = st.sidebar.file_uploader("Upload GeoTIFF (.tif)", type=["tif", "tiff"])

raster_path = None
raster_footprint = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        raster_path = tmp.name
    st.sidebar.success("Raster loaded from upload")
    # Get footprint for map
    raster_footprint = get_raster_footprint_4326(raster_path)
else:
    DEFAULT_RASTER_PATH = "soil_moisture.tif"
    if os.path.exists(DEFAULT_RASTER_PATH):
        raster_path = DEFAULT_RASTER_PATH
        st.sidebar.success(f"Using default raster: {DEFAULT_RASTER_PATH}")
        raster_footprint = get_raster_footprint_4326(raster_path)
    else:
        st.sidebar.error("No raster file. Please upload a GeoTIFF or place 'soil_moisture.tif' in the app directory.")

# ---------- Map with drawing and raster footprint ----------
st.sidebar.markdown("---")
st.sidebar.markdown("## 🗺️ Draw Site Polygon")
st.sidebar.info("Use the polygon tool on the map (top‑right). Double‑click to finish.")

# Center map on Ireland or raster footprint
if raster_footprint is not None:
    center_lat = raster_footprint.centroid.y
    center_lon = raster_footprint.centroid.x
else:
    center_lat, center_lon = 53.3498, -6.2603

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# Draw raster footprint if available
if raster_footprint is not None:
    folium.GeoJson(
        data=raster_footprint.__geo_interface__,
        style_function=lambda x: {'color': 'red', 'weight': 2, 'fillOpacity': 0.1, 'fillColor': 'red'},
        name="Raster Footprint"
    ).add_to(m)

draw = Draw(
    draw_options={
        "polygon": {"allowIntersection": False, "showArea": True},
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "marker": False,
    },
    edit_options={"edit": True}
)
draw.add_to(m)
map_output = st_folium(m, width=800, height=500)

# Extract polygon coordinates (WGS84)
polygon_coords_4326 = None
if map_output and map_output.get("last_active_drawing"):
    geom = map_output["last_active_drawing"]["geometry"]
    if geom["type"] == "Polygon":
        polygon_coords_4326 = geom["coordinates"][0]
        # Compute centroid for display
        lats = [p[1] for p in polygon_coords_4326]
        lngs = [p[0] for p in polygon_coords_4326]
        clat, clng = sum(lats)/len(lats), sum(lngs)/len(lngs)
        st.sidebar.success(f"Polygon centroid: {clat:.5f}, {clng:.5f}")
    else:
        st.sidebar.warning("Draw a polygon (not a line/point)")

# ---------- Manual override ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Manual Override (optional)")
use_manual = st.sidebar.checkbox("Use custom bounding box", value=False)
if use_manual:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        minx = st.number_input("West (min lon)", value=-6.35, format="%.5f")
        miny = st.number_input("South (min lat)", value=53.30, format="%.5f")
    with col2:
        maxx = st.number_input("East (max lon)", value=-6.20, format="%.5f")
        maxy = st.number_input("North (max lat)", value=53.40, format="%.5f")
    polygon_coords_4326 = [
        [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
    ]
    st.sidebar.write("Using rectangle bbox (WGS84)")

# ---------- Run analysis ----------
if st.sidebar.button("🚀 Extract Soil Moisture", type="primary", use_container_width=True):
    if raster_path is None:
        st.error("No raster file available. Please upload a GeoTIFF.")
    elif polygon_coords_4326 is None:
        st.error("Please draw a polygon on the map or enable manual bounding box.")
    else:
        with st.spinner("Clipping raster..."):
            img, mean_val, min_val, max_val = clip_raster_with_polygon(raster_path, polygon_coords_4326)
            if img is not None:
                st.session_state['clipped_img'] = img
                st.session_state['mean_ssm'] = mean_val
                st.session_state['min_ssm'] = min_val
                st.session_state['max_ssm'] = max_val
                st.success("Extraction complete!")

# ---------- Display results ----------
if 'clipped_img' in st.session_state:
    img = st.session_state['clipped_img']
    mean_val = st.session_state['mean_ssm']
    min_val = st.session_state['min_ssm']
    max_val = st.session_state['max_ssm']
    
    st.markdown("---")
    st.subheader("📊 Soil Moisture Index")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Value", f"{mean_val:.4f} m³/m³")
    col2.metric("Minimum", f"{min_val:.4f} m³/m³")
    col3.metric("Maximum", f"{max_val:.4f} m³/m³")
    
    # Risk classification
    if mean_val > 0.30:
        risk = "HIGH"
        condition = "BRE365 soakaway test mandatory; infiltration unlikely."
        color = "red"
    elif mean_val > 0.20:
        risk = "MEDIUM"
        condition = "BRE365 soakaway test recommended."
        color = "orange"
    else:
        risk = "LOW"
        condition = "Standard surface water drainage acceptable."
        color = "green"
    
    st.markdown(f"### Risk Level: <span style='color:{color}; font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
    st.info(f"📌 **Draft Planning Condition:** {condition}")
    
    # Visualise clipped raster
    st.subheader("🗺️ Clipped Soil Moisture Layer")
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.Blues
    im = ax.imshow(img, cmap=cmap, norm=Normalize(vmin=min_val, vmax=max_val))
    ax.set_title("Soil Moisture (m³/m³) over selected area")
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Soil moisture (m³/m³)")
    st.pyplot(fig)
    
    st.caption("Data: Local GeoTIFF (Copernicus CLMS).")
else:
    st.info("👈 Draw a polygon on the map, then click 'Extract Soil Moisture'.")
