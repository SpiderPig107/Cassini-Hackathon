import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import pandas as pd
import numpy as np
from shapely.geometry import shape, polygon
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="AquaScan | Planning Decision Support")

# Custom CSS for LPA-style Dashboard
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .report-section { background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALISATION ---
if 'assessment_run' not in st.session_state:
    st.session_state.assessment_run = False
if 'site_area' not in st.session_state:
    st.session_state.site_area = 0.0
if 'coords' not in st.session_state:
    st.session_state.coords = (53.3498, -6.2603)

# --- SIDEBAR: SITE REGISTRATION (PRD 4.1) ---
with st.sidebar:
    st.title("🛰️ AquaScan MVP")
    st.caption("v0.1 | Dublin Pilot")
    
    with st.expander("Administrative Details", expanded=True):
        app_ref = st.text_input("Planning Ref", value="AQS-2026-0001")
        lpa_choice = st.selectbox("Local Authority", ["Dublin City Council", "Fingal", "South Dublin", "DLR"])
        unit_count = st.number_input("Proposed Units", min_value=1, value=50)

    with st.expander("GNSS & Precision (Galileo/EGNOS)", expanded=True):
        egnos_toggle = st.toggle("EGNOS Augmentation Active", value=True)
        st.info(f"Target Accuracy: {'±0.8m' if egnos_toggle else '±3.0m'}")
        
    st.divider()
    # Dynamic Inputs from Map
    st.write("**Derived Site Data**")
    lat_input = st.number_input("Latitude", value=st.session_state.coords[0], format="%.6f")
    lng_input = st.number_input("Longitude", value=st.session_state.coords[1], format="%.6f")
    area_input = st.number_input("Site Area (ha)", value=st.session_state.site_area, format="%.2f")
    
    if st.button("🚀 Run Assessment"):
        if area_input <= 0:
            st.error("Please draw a site boundary first.")
        else:
            st.session_state.assessment_run = True

# --- MAIN INTERFACE ---
st.title(f"Planning Assessment Report: {app_ref}")
st.write(f"**LPA:** {lpa_choice} | **User:** Planning Officer | **Status:** { 'Complete' if st.session_state.assessment_run else 'Pending Site Delineation' }")

# Map Section
col_map, col_info = st.columns([2, 1])

with col_map:
    m = folium.Map(location=[53.3498, -6.2603], zoom_start=14, tiles="CartoDB Positron")
    draw = Draw(
        export=False,
        draw_options={'polyline': False, 'rectangle': True, 'polygon': True, 'circle': False, 'marker': False},
        edit_options={'edit': True}
    )
    draw.add_to(m)
    
    # Catching map interactions
    map_output = st_folium(m, width="100%", height=500)

    # Logic to update Sidebar based on Map Drawing
    if map_output['last_active_drawing']:
        geometry = map_output['last_active_drawing']['geometry']
        s = shape(geometry)
        
        # Calculate Area (Approximate for WGS84)
        # In production, we'd project to EPSG:2157 (ITM) for accuracy
        area_sqm = s.area * (111000 * 111000 * np.cos(np.radians(53.3)))
        st.session_state.site_area = round(area_sqm / 10000, 2)
        
        # Get Centroid Coords
        centroid = s.centroid
        st.session_state.coords = (centroid.y, centroid.x)

with col_info:
    st.subheader("Site Overview")
    st.markdown(f"""
    - **Galileo Lat:** `{st.session_state.coords[0]:.6f}`
    - **Galileo Lng:** `{st.session_state.coords[1]:.6f}`
    - **Total Area:** `{st.session_state.site_area} ha`
    - **EGNOS Status:** {'✅ Corrected' if egnos_toggle else '⚠️ Standard'}
    """)
    if st.session_state.site_area > 50:
        st.warning("Flag: Site exceeds 50 ha. Manual review required.")

# --- ASSESSMENT RESULTS (PRD 4.2 - 4.6) ---
if st.session_state.assessment_run:
    st.divider()
    
    # Simulation Logic for Modules
    # In a real environment, these call Copernicus APIs
    modules = {
        "Flood Zone": {"val": "Zone B", "risk": "MEDIUM", "ref": "OPW 2009 Guidelines"},
        "Imperviousness": {"val": "Delta +42%", "risk": "HIGH", "ref": "GDSDS 2005"},
        "Soil Moisture": {"val": "SWI 78", "risk": "HIGH", "ref": "BRE Digest 365"},
        "Ground Motion": {"val": "-0.5mm/yr", "risk": "LOW", "ref": "EGMS Ortho"},
        "WFD Status": {"val": "Moderate", "risk": "MEDIUM", "ref": "SI 272/2009"}
    }

    cols = st.columns(5)
    for i, (name, data) in enumerate(modules.items()):
        with cols[i]:
            st.metric(name, data["val"], data["risk"], delta_color="inverse" if data["risk"] == "HIGH" else "normal")
            st.caption(f"Ref: {data['ref']}")

    # --- SUMMARY SECTION (PRD 4.7) ---
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.subheader("Executive Summary & Determination")
    
    risk_col, rec_col = st.columns(2)
    risk_col.error("OVERALL RISK: HIGH (Triggered by Soil Saturation & Runoff Delta)")
    rec_col.warning("RECOMMENDATION: Further Information Required (FI Prescribed)")

    st.write("---")
    st.subheader("Draft Planning Conditions")
    
    cond1 = st.text_area("Condition 1: Flood Risk", 
        "The applicant shall submit a site-specific Flood Risk Assessment (Stage 3) including a Justification Test, in accordance with OPW Guidelines 2009.")
    
    cond2 = st.text_area("Condition 2: SuDS Attenuation", 
        f"Attenuation storage must be provided for the {unit_count} units to limit discharge to 2 l/s/ha for the 1-in-100 year event.")
    
    cond3 = st.text_area("Condition 3: Soil Testing", 
        "Due to high satellite-derived Soil Water Index (SWI 78), field infiltration testing per BRE Digest 365 is mandatory prior to commencement.")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- HYDROLOGICAL SCIENCE (FORMULA REQUIREMENT) ---
    st.divider()
    st.write("**The Attenuation Formula (Layman's Version):**")
    st.info("Peak Flow = (Hardness of Surface) × (Rainfall Intensity) × (Size of Site)")

    st.write("**The Scientific Notation (Rational Method):**")
    st.latex(r"Q = C \cdot i \cdot A")

    with st.expander("Variable Definitions"):
        st.write("""
        | Symbol | Definition |
        | :--- | :--- |
        | **Q** | Peak Rate of Runoff (m³/s). |
        | **C** | **Runoff Coefficient.** Value between 0-1 based on site imperviousness. |
        | **i** | Rainfall Intensity (mm/hr) based on Met Éireann data. |
        | **A** | Area of the site in hectares. |
        """)

    # Footer/Export
    st.divider()
    col_dl, col_attr = st.columns([1, 1])
    col_dl.button("📥 Download PDF Assessment (AQS-2026-0001)")
    col_attr.caption(f"Data attribution: Copernicus CLMS (2018), EGMS (20
