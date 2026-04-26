import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tempfile, os, time
import openeo
import openeo.processes
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import date, timedelta

st.set_page_config(page_title="AquaScan - Sentinel-1 Soil Moisture", layout="wide")
st.title("🌍 AquaScan – Surface Soil Moisture from Sentinel‑1")
st.markdown("Draw a polygon → fetch pre-processed SSM from Copernicus Land Monitoring Service")

# ── Constants ────────────────────────────────────────────────────────────────
MAX_SYNC_DEG2 = 0.25   # AOIs larger than this go to batch mode

# CGLS_SSM_V1_EUROPE coverage: Europe only, 1 km resolution, daily, from ~2015
CGLS_COLLECTION   = "CGLS_SSM_V1_EUROPE"
CGLS_BAND         = "SSM"       # 0–100 scale (dry→wet)
FALLBACK_ENABLED  = True        # fall back to computed GRD if CGLS fails
FALLBACK_RES_M    = 160         # resolution for the GRD fallback pipeline

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_bbox(coords):
    lngs = [p[0] for p in coords]; lats = [p[1] for p in coords]
    return min(lngs), min(lats), max(lngs), max(lats)

def run_batch_with_progress(job, outfile):
    """Poll a batch job and stream status updates into the Streamlit UI."""
    status_box   = st.empty()
    progress_bar = st.progress(0)
    WEIGHTS = {"created": 0.05, "queued": 0.10, "running": 0.60,
               "finished": 1.0, "error": 1.0, "canceled": 1.0}
    while True:
        info   = job.status()
        status = info if isinstance(info, str) else info.get("status", "unknown")
        progress_bar.progress(WEIGHTS.get(status, 0.0))
        status_box.info(
            f"⏳ Batch job: **{status}** — "
            f"[CDSE dashboard](https://dataspace.copernicus.eu/analyse/openeo)"
        )
        if status == "finished":
            break
        if status in ("error", "canceled"):
            raise RuntimeError(f"openEO batch job failed with status: {status}")
        time.sleep(15)
    job.get_results().download_files(outfile)
    status_box.success("✅ Done!")
    progress_bar.progress(1.0)

# ── Connection ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Copernicus Data Space…")
def get_connection():
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()
    return conn

# ── PRIMARY: fetch pre-processed CGLS SSM ───────────────────────────────────
# CGLS_SSM_V1_EUROPE is the Copernicus Land Monitoring Service product that
# already runs the full change-detection / normalisation pipeline server-side.
# Loading it skips sar_backscatter() entirely — no terrain correction, no
# reference stack computation. The result is identical science at a fraction
# of the processing cost. Values are 0–100; we rescale to 0–1 for display.
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cgls_ssm(west, south, east, north, date_start, date_end):
    conn   = get_connection()
    extent = {"west": west, "south": south, "east": east, "north": north}

    cube = conn.load_collection(
        CGLS_COLLECTION,
        spatial_extent=extent,
        temporal_extent=[date_start, date_end],
        bands=[CGLS_BAND],
    )
    # Take the most recent valid observation in the window
    cube = cube.reduce_dimension(dimension="t", reducer="last")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        outfile = tmp.name

    area = (east - west) * (north - south)
    if area <= MAX_SYNC_DEG2:
        cube.download(outfile, format="NetCDF")
    else:
        job = cube.execute_batch(title="AquaScan_CGLS_SSM")
        run_batch_with_progress(job, outfile)

    ds = xr.load_dataset(outfile)
    os.unlink(outfile)
    return ds, "cgls"

# ── FALLBACK: compute SSM from raw Sentinel-1 GRD ───────────────────────────
# Used only when CGLS fails (e.g. collection unavailable, API error) or when
# the AOI is outside Europe. Significantly slower due to sar_backscatter().
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_computed_ssm(west, south, east, north, date_start, date_end):
    conn   = get_connection()
    extent = {"west": west, "south": south, "east": east, "north": north}

    # Reference window: 6 months before the query start
    ref_start = (date.fromisoformat(date_start) - timedelta(days=180)).isoformat()

    def load_grd(t_start, t_end):
        cube = conn.load_collection(
            "SENTINEL1_GRD",
            temporal_extent=[t_start, t_end],
            spatial_extent=extent,
            bands=["VV"],
        )
        cube = cube.resample_spatial(resolution=FALLBACK_RES_M, method="average")
        cube = cube.sar_backscatter(coefficient="sigma0-ellipsoid")
        return cube

    s1_ref = load_grd(ref_start, date_start)
    s1_cur = load_grd(date_start, date_end)

    s1_cur  = s1_cur.reduce_dimension(dimension="t", reducer="last")
    dry_ref = s1_ref.reduce_dimension(dimension="t", reducer="min")
    wet_ref = s1_ref.reduce_dimension(dimension="t", reducer="max")
    SSM     = (s1_cur - dry_ref) / (wet_ref - dry_ref)

    avg_ref = s1_ref.reduce_dimension(dimension="t", reducer="mean")
    avg_ref = avg_ref.apply(lambda d: 10 * openeo.processes.log(d, base=10))
    mask    = (avg_ref.band("VV") > -6) | (avg_ref.band("VV") < -17)
    SSM     = SSM.mask(mask)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        outfile = tmp.name

    area = (east - west) * (north - south)
    if area <= MAX_SYNC_DEG2:
        SSM.download(outfile, format="NetCDF")
    else:
        job = SSM.execute_batch(title="AquaScan_GRD_SSM")
        run_batch_with_progress(job, outfile)

    ds = xr.load_dataset(outfile)
    os.unlink(outfile)
    return ds, "computed"

# ── Sidebar: draw + coordinates ──────────────────────────────────────────────
st.sidebar.header("🗺️ Draw your area of interest")
st.sidebar.info("Use the polygon tool (top-right on map). Double-click to finish.")

m = folium.Map(location=[53.3498, -6.2603], zoom_start=7, tiles="CartoDB positron")
Draw(
    draw_options={"polygon": {"allowIntersection": False, "showArea": True},
                  "polyline": False, "rectangle": False, "circle": False, "marker": False},
    edit_options={"edit": True},
).add_to(m)

map_output = st_folium(
    m, width=800, height=500,
    key="main_map",
    returned_objects=["last_active_drawing"],
)

bbox = None
if map_output and map_output.get("last_active_drawing"):
    geom = map_output["last_active_drawing"]["geometry"]
    if geom["type"] == "Polygon":
        bbox = get_bbox(geom["coordinates"][0])
        st.sidebar.success(f"BBox: {[round(v,4) for v in bbox]}")
    else:
        st.sidebar.warning("Draw a polygon, not a line/point.")

st.sidebar.markdown("---")
st.sidebar.subheader("Coordinates")
_b = bbox or (-6.35, 53.30, -6.20, 53.40)
c1, c2 = st.sidebar.columns(2)
with c1:
    west_m  = st.number_input("West",  value=_b[0], format="%.4f")
    south_m = st.number_input("South", value=_b[1], format="%.4f")
with c2:
    east_m  = st.number_input("East",  value=_b[2], format="%.4f")
    north_m = st.number_input("North", value=_b[3], format="%.4f")

use_manual = st.sidebar.checkbox("Override with manual coordinates", value=(bbox is None))
if use_manual:
    spatial_extent = {"west": west_m, "south": south_m, "east": east_m, "north": north_m}
elif bbox:
    spatial_extent = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}
else:
    spatial_extent = None

# ── Sidebar: date range ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Date range")
# CGLS archive goes back to ~2015; default to last 30 days
default_end   = date.today()
default_start = default_end - timedelta(days=30)
date_start = st.sidebar.date_input("From", value=default_start, min_value=date(2015, 1, 1))
date_end   = st.sidebar.date_input("To",   value=default_end,   min_value=date(2015, 1, 2))
if date_start >= date_end:
    st.sidebar.error("'From' must be before 'To'.")

# ── Sidebar: data source selector ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
source_choice = st.sidebar.radio(
    "Source",
    ["🚀 Pre-processed CGLS (fast, 1 km, Europe only)",
     "🔬 Compute from Sentinel-1 GRD (slow, higher res)"],
    index=0,
)
use_cgls = source_choice.startswith("🚀")

if spatial_extent:
    area_deg2 = (spatial_extent["east"] - spatial_extent["west"]) * \
                (spatial_extent["north"] - spatial_extent["south"])
    mode = "sync" if area_deg2 <= MAX_SYNC_DEG2 else "batch"
    st.sidebar.caption(
        f"AOI: {area_deg2:.4f}°²  ·  mode: **{mode}**"
        + ("  ·  ~seconds" if mode == "sync" else "  ·  ~10–30 min")
    )

# ── Compute ──────────────────────────────────────────────────────────────────
if st.sidebar.button("🚀 Fetch Soil Moisture", type="primary", use_container_width=True):
    if not spatial_extent:
        st.error("Please draw a polygon or enter coordinates first.")
    elif date_start >= date_end:
        st.error("Fix the date range before running.")
    else:
        with st.spinner("Fetching from Copernicus Data Space…"):
            try:
                if use_cgls:
                    ds, source = fetch_cgls_ssm(
                        spatial_extent["west"], spatial_extent["south"],
                        spatial_extent["east"], spatial_extent["north"],
                        date_start.isoformat(), date_end.isoformat(),
                    )
                else:
                    ds, source = fetch_computed_ssm(
                        spatial_extent["west"], spatial_extent["south"],
                        spatial_extent["east"], spatial_extent["north"],
                        date_start.isoformat(), date_end.isoformat(),
                    )
                st.session_state["ds"]     = ds
                st.session_state["source"] = source
                st.success("Done!")
                st.rerun()
            except Exception as e:
                if use_cgls and FALLBACK_ENABLED:
                    st.warning(f"CGLS fetch failed ({e}). Falling back to computed GRD pipeline…")
                    try:
                        ds, source = fetch_computed_ssm(
                            spatial_extent["west"], spatial_extent["south"],
                            spatial_extent["east"], spatial_extent["north"],
                            date_start.isoformat(), date_end.isoformat(),
                        )
                        st.session_state["ds"]     = ds
                        st.session_state["source"] = source
                        st.success("Done (via fallback pipeline)!")
                        st.rerun()
                    except Exception as e2:
                        st.error(f"Fallback also failed: {e2}")
                else:
                    st.error(f"Error: {e}")
                    st.info("Check terminal for the device-code auth URL on first run.")

# ── Results ──────────────────────────────────────────────────────────────────
if "ds" in st.session_state:
    ds     = st.session_state["ds"]
    source = st.session_state.get("source", "unknown")
    data_vars = list(ds.data_vars)

    if not data_vars:
        st.error("No data variable found in result.")
    else:
        raw  = ds[data_vars[0]]

        # CGLS is 0–100; normalise to 0–1 for consistent display and metrics.
        # Computed GRD pipeline already outputs 0–1.
        if source == "cgls":
            data = raw / 100.0
            res_label = "1 km (CGLS pre-processed)"
        else:
            data = raw
            res_label = f"{FALLBACK_RES_M} m (computed from GRD)"

        # Squeeze any leftover length-1 dimensions (e.g. a residual t or band axis)
        data = data.squeeze()

        col_map, col_stats = st.columns([3, 1])

        with col_map:
            fig, ax = plt.subplots(figsize=(9, 7))
            im = data.plot.imshow(
                ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                add_colorbar=False,
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Soil Moisture (0 = dry, 1 = wet)")
            # Discrete tick labels
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(["Dry", "Low", "Medium", "High", "Wet"])
            ax.set_title(
                f"Surface Soil Moisture  ·  {date_start} → {date_end}\n"
                f"Source: {'CGLS (pre-processed)' if source == 'cgls' else 'Computed GRD'}  ·  {res_label}"
            )
            st.pyplot(fig)
            plt.close(fig)

        with col_stats:
            vals = data.values.ravel()
            vals = vals[~np.isnan(vals)]

            st.metric("Mean SSM",   f"{vals.mean():.2f}")
            st.metric("Median SSM", f"{np.median(vals):.2f}")
            st.metric("Std dev",    f"{vals.std():.2f}")
            st.metric("Valid px",   f"{len(vals):,}")

            # Moisture category breakdown
            st.markdown("**Moisture categories**")
            cats = {
                "🟤 Dry (<0.25)":       (vals < 0.25).sum(),
                "🟡 Low (0.25–0.5)":    ((vals >= 0.25) & (vals < 0.5)).sum(),
                "🟢 Medium (0.5–0.75)": ((vals >= 0.5)  & (vals < 0.75)).sum(),
                "💧 Wet (>0.75)":       (vals >= 0.75).sum(),
            }
            for label, count in cats.items():
                pct = 100 * count / len(vals) if len(vals) else 0
                st.write(f"{label}: **{pct:.1f}%**")

            st.caption(
                "Source: CGLS Surface Soil Moisture V1 (Sentinel-1 based)\n\n"
                if source == "cgls" else
                "Source: Computed from Sentinel-1 GRD (sigma0-ellipsoid)\n\n"
            )
else:
    st.info("👈 Draw a polygon and click 'Fetch Soil Moisture'.")
