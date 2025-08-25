import streamlit as st 
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import folium
from streamlit_folium import st_folium
import json
import warnings
warnings.filterwarnings('ignore')

# =============================
# 🔐 API Key & LLM Utilities
# =============================

def get_env_or_secret(key_name: str, default: str = None):
    """Helper to read from Streamlit secrets first, then env vars."""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
        # Fall back to environment variables
        return os.getenv(key_name, default)
    except Exception:
        return os.getenv(key_name, default)


def get_active_llm_provider():
    """Determine which LLM provider is available (AI/ML API preferred, then Groq)."""
    ai_ml_key = get_env_or_secret("AI_ML_API_KEY")
    groq_key = get_env_or_secret("GROQ_API_KEY")
    if ai_ml_key:
        return "ai_ml"
    if groq_key:
        return "groq"
    return None


# =============================
# 🔐 AIML API Integration (robust)
# =============================

def get_llm_summary(prompt: str, context: str = "") -> str:
    """
    Robust LLM summary using AIML /responses endpoint.
    """
    # Build final prompt safely
    if context:
        full_prompt = f"{context}\nUser Query: {prompt}"
    else:
        full_prompt = prompt

    api_key = get_env_or_secret("AI_ML_API_KEY")
    if not api_key:
        return (
            "AI Analysis unavailable — AI_ML_API_KEY not configured.\n"
            "Go to Settings → Secrets → Create secret 'AI_ML_API_KEY' with your AI/ML API key."
        )

    url = "https://api.aimlapi.com/v1/responses"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def call_api(payload):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code not in (200, 201):
                return None, f"AI Analysis Error (AIML API {r.status_code}): {r.text}"
            return r.json(), None
        except requests.exceptions.RequestException as e:
            return None, f"AI Analysis Request Failed (network): {e}"
        except Exception as e:
            return None, f"AI Analysis Request Failed (parsing): {e}"

    # Fixed payload with correct model name and input format
    base_payload = {
        "model": "gpt-4o",  # Changed from "openai/gpt-4" to "gpt-4o"
        "input": full_prompt,  # Changed from {"text": full_prompt} to just full_prompt
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "response_format": {"type": "text"}
    }

    data, error = call_api(base_payload)

    # If API complains about unsupported 'temperature', remove it and retry once
    if error and "Unsupported parameter" in error and "temperature" in error:
        payload_no_temp = base_payload.copy()
        payload_no_temp.pop("temperature", None)
        data, error = call_api(payload_no_temp)

    if error:
        return error

    def extract_text_from_response(data_dict):
        texts = []

        # Top-level output_text
        ot = data_dict.get("output_text")
        if isinstance(ot, str) and ot.strip():
            texts.append(ot.strip())

        # Walk `output` items
        for item in data_dict.get("output", []) or []:
            itype = item.get("type", "")

            # Reasoning summaries (collect if present)
            if itype == "reasoning":
                for s in item.get("summary", []) or []:
                    if isinstance(s, dict):
                        t = s.get("text") or s.get("summary") or ""
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())

            # Message-like items with content list
            if itype == "message" and isinstance(item.get("content"), list):
                for c in item["content"]:
                    if isinstance(c, dict):
                        txt = c.get("text") or c.get("output_text") or ""
                        if isinstance(txt, str) and txt.strip():
                            texts.append(txt.strip())

            # Defensive: direct 'text' field
            if isinstance(item.get("text"), str) and item.get("text").strip():
                texts.append(item.get("text").strip())

        # Fallback: 'responses' list
        if not texts and isinstance(data_dict.get("responses"), list):
            for r in data_dict.get("responses"):
                if isinstance(r, dict):
                    for f in ("output_text", "text"):
                        v = r.get(f)
                        if isinstance(v, str) and v.strip():
                            texts.append(v.strip())
                    cont = r.get("content")
                    if isinstance(cont, list):
                        for c in cont:
                            if isinstance(c, dict) and c.get("text"):
                                texts.append(c["text"].strip())

        return "\n".join(texts).strip()

    out_text = extract_text_from_response(data)
    # If we received only reasoning without text, try an explicit retry that forces text
    if not out_text:
        # Second attempt: try with gpt-4o-mini as fallback
        retry_payload = {
            "model": "gpt-4o-mini",  # Use mini version as fallback
            "input": full_prompt,
            "max_output_tokens": 2048,
            "response_format": {"type": "text"}
        }
        data2, err2 = call_api(retry_payload)
        if err2:
            return err2
        out_text2 = extract_text_from_response(data2)
        if out_text2:
            return out_text2

    if out_text:
        return out_text

    # Final fallback: return debug excerpt for inspection
    try:
        return "AI Analysis returned no text output. Raw response excerpt: " + json.dumps(data)[:1600]
    except Exception:
        return "AI Analysis returned no text output and response could not be serialized."


# =============================
# 🎨 UI Color Schemes & Risk Tables
# =============================

MAGNITUDE_COLORS = {
    'Low': '#00ff00',
    'Moderate': '#ffff00',
    'High': '#ff8000',
    'Severe': '#ff0000',
    'Extreme': '#800000'
}

RISK_THRESHOLDS = {
    'low': {'count': 5, 'max_magnitude': 3.0},
    'moderate': {'count': 10, 'max_magnitude': 4.5},
    'high': {'count': 20, 'max_magnitude': 5.5},
    'severe': {'count': 30, 'max_magnitude': 6.5},
    'extreme': {'count': 50, 'max_magnitude': 7.0}
}

EMERGENCY_PROTOCOLS = {
    'low': "Monitor situation. No immediate action required.",
    'moderate': "Stay alert. Review emergency plans.",
    'high': "Prepare emergency kit. Stay informed.",
    'severe': "Follow evacuation orders if issued. Seek shelter.",
    'extreme': "IMMEDIATE EVACUATION. Follow emergency services."
}


# =============================
# 🌐 Data Fetching & Processing
# =============================

def fetch_earthquakes(min_magnitude=2.5, hours=24, region_bbox=None, detailed=True):
    try:
        endtime = datetime.utcnow()
        starttime = endtime - timedelta(hours=hours)

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": starttime.strftime('%Y-%m-%dT%H:%M:%S'),
            "endtime": endtime.strftime('%Y-%m-%dT%H:%M:%S'),
            "minmagnitude": min_magnitude,
            "orderby": "time",
            "limit": 500 if detailed else 200
        }

        if region_bbox:
            params.update({
                "minlatitude": region_bbox[1],
                "maxlatitude": region_bbox[3],
                "minlongitude": region_bbox[0],
                "maxlongitude": region_bbox[2],
            })

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        features = data.get('features', [])
        earthquakes = []

        for f in features:
            prop = f['properties']
            geom = f['geometry']
            earthquake = {
                'time': datetime.utcfromtimestamp(prop['time']/1000),
                'place': prop.get('place', 'Unknown'),
                'magnitude': prop.get('mag'),
                'longitude': geom['coordinates'][0],
                'latitude': geom['coordinates'][1],
                'depth': geom['coordinates'][2],
                'url': prop.get('url', ''),
                'type': prop.get('type', 'earthquake'),
                'status': prop.get('status', 'automatic'),
                'tsunami': prop.get('tsunami', 0),
                'felt': prop.get('felt', 0),
                'cdi': prop.get('cdi', 0),
                'mmi': prop.get('mmi', 0),
                'alert': prop.get('alert', ''),
                'sig': prop.get('sig', 0)
            }
            earthquake['risk_level'] = calculate_risk_level(earthquake['magnitude'])
            earthquake['time_ago'] = calculate_time_ago(earthquake['time'])
            earthquakes.append(earthquake)

        df = pd.DataFrame(earthquakes)
        if not df.empty:
            df['magnitude_category'] = df['magnitude'].apply(categorize_magnitude)
            df['depth_category'] = df['depth'].apply(categorize_depth)
            df['hour_of_day'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.day_name()
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data processing error: {e}")
        return pd.DataFrame()


# =============================
# 📊 Analytics Helpers
# =============================

def calculate_risk_level(magnitude):
    if magnitude is None:
        return 'Low'
    if magnitude >= 7.0:
        return 'Extreme'
    elif magnitude >= 6.0:
        return 'Severe'
    elif magnitude >= 5.0:
        return 'High'
    elif magnitude >= 4.0:
        return 'Moderate'
    else:
        return 'Low'


def categorize_magnitude(magnitude):
    if magnitude is None:
        return 'Unknown'
    if magnitude >= 7.0:
        return 'Major (≥7.0)'
    elif magnitude >= 6.0:
        return 'Strong (6.0-6.9)'
    elif magnitude >= 5.0:
        return 'Moderate (5.0-5.9)'
    elif magnitude >= 4.0:
        return 'Light (4.0-4.9)'
    else:
        return 'Minor (<4.0)'


def categorize_depth(depth):
    if pd.isna(depth):
        return 'Unknown'
    if depth < 70:
        return 'Shallow (<70km)'
    elif depth < 300:
        return 'Intermediate (70-300km)'
    else:
        return 'Deep (>300km)'


def calculate_time_ago(time_val: datetime):
    now = datetime.utcnow()
    diff = now - time_val
    if diff.days > 0:
        return f"{diff.days} day(s) ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour(s) ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute(s) ago"
    else:
        return "Just now"


def analyze_seismic_patterns(df: pd.DataFrame):
    if df.empty:
        return {}
    analysis = {}
    try:
        if len(df) > 0:
            analysis['hourly_distribution'] = df['hour_of_day'].value_counts().sort_index()
            analysis['daily_distribution'] = df['day_of_week'].value_counts()
        if 'magnitude' in df.columns and len(df) > 0:
            analysis['magnitude_stats'] = {
                'mean': df['magnitude'].mean(),
                'median': df['magnitude'].median(),
                'std': df['magnitude'].std(),
                'max': df['magnitude'].max(),
                'min': df['magnitude'].min()
            }
        if 'depth' in df.columns and len(df) > 0:
            analysis['depth_stats'] = {
                'mean': df['depth'].mean(),
                'median': df['depth'].median(),
                'std': df['depth'].std()
            }
        if 'risk_level' in df.columns and len(df) > 0:
            analysis['risk_distribution'] = df['risk_level'].value_counts()
        if len(df) > 1 and 'latitude' in df.columns and 'longitude' in df.columns:
            analysis['geographic_center'] = {
                'lat': df['latitude'].mean(),
                'lon': df['longitude'].mean()
            }
    except Exception as e:
        st.warning(f"Error in pattern analysis: {str(e)}")
        return {}
    return analysis


def calculate_overall_risk(df: pd.DataFrame):
    if df.empty:
        return 'low', "Risk Score: 0/80"
    count = len(df)
    max_magnitude = df['magnitude'].max()
    risk_score = 0
    if count >= RISK_THRESHOLDS['extreme']['count']:
        risk_score += 40
    elif count >= RISK_THRESHOLDS['severe']['count']:
        risk_score += 30
    elif count >= RISK_THRESHOLDS['high']['count']:
        risk_score += 20
    elif count >= RISK_THRESHOLDS['moderate']['count']:
        risk_score += 10
    if max_magnitude >= RISK_THRESHOLDS['extreme']['max_magnitude']:
        risk_score += 40
    elif max_magnitude >= RISK_THRESHOLDS['severe']['max_magnitude']:
        risk_score += 30
    elif max_magnitude >= RISK_THRESHOLDS['high']['max_magnitude']:
        risk_score += 20
    elif max_magnitude >= RISK_THRESHOLDS['moderate']['max_magnitude']:
        risk_score += 10
    if risk_score >= 60:
        risk_level = 'extreme'
    elif risk_score >= 40:
        risk_level = 'severe'
    elif risk_score >= 25:
        risk_level = 'high'
    elif risk_score >= 10:
        risk_level = 'moderate'
    else:
        risk_level = 'low'
    return risk_level, f"Risk Score: {risk_score}/80"


# =============================
# 🗺️ Visualization Helpers
# =============================

def create_advanced_map(df: pd.DataFrame, region_bbox=None):
    if df.empty:
        return None
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='OpenStreetMap')
    for _, row in df.iterrows():
        if row['magnitude'] >= 6.0:
            color = 'red'
            radius = 15
        elif row['magnitude'] >= 5.0:
            color = 'orange'
            radius = 12
        elif row['magnitude'] >= 4.0:
            color = 'yellow'
            radius = 10
        else:
            color = 'green'
            radius = 8
        popup_content = (
            f"<b>Magnitude {row['magnitude']}</b><br>"
            f"Location: {row['place']}<br>"
            f"Time: {row['time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"Depth: {row['depth']:.1f} km<br>"
            f"<a href=\"{row['url']}\" target=\"_blank\">USGS Details</a>"
        )
        folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=radius,
                            popup=popup_content, color=color, fill=True, fillOpacity=0.7).add_to(m)
    if region_bbox:
        folium.Rectangle(bounds=[[region_bbox[1], region_bbox[0]], [region_bbox[3], region_bbox[2]]],
                         color='blue', weight=2, fillOpacity=0.1).add_to(m)
    return m


def create_comprehensive_charts(df: pd.DataFrame, analysis: dict):
    if df.empty:
        return []
    charts = []
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['time'], y=df['magnitude'], mode='markers',
                              marker=dict(size=df['magnitude'] * 2, color=df['magnitude'], colorscale='Reds', showscale=True),
                              name='Earthquakes'))
    if len(df) >= 2:
        try:
            z = np.polyfit(range(len(df)), df['magnitude'], 1)
            p = np.poly1d(z)
            fig1.add_trace(go.Scatter(x=df['time'], y=p(range(len(df))), mode='lines', name='Trend', line=dict(color='blue', dash='dash')))
        except (np.linalg.LinAlgError, ValueError) as e:
            st.warning(f"Trend analysis unavailable: {str(e)}")
    fig1.update_layout(title='Earthquake Magnitude Over Time with Trend', xaxis_title='Time', yaxis_title='Magnitude', height=400)
    charts.append(fig1)
    if len(df) > 0:
        fig2 = px.histogram(df, x='magnitude', nbins=min(20, len(df)), title='Magnitude Distribution', labels={'magnitude': 'Magnitude', 'count': 'Frequency'})
        fig2.update_layout(height=400)
        charts.append(fig2)
        fig3 = px.scatter(df, x='depth', y='magnitude', color='magnitude', title='Depth vs Magnitude Relationship', labels={'depth': 'Depth (km)', 'magnitude': 'Magnitude'})
        fig3.update_layout(height=400)
        charts.append(fig3)
    if 'hourly_distribution' in analysis and len(analysis['hourly_distribution']) > 0:
        fig4 = px.bar(x=analysis['hourly_distribution'].index, y=analysis['hourly_distribution'].values, title='Earthquake Activity by Hour of Day', labels={'x': 'Hour', 'y': 'Count'})
        fig4.update_layout(height=400)
        charts.append(fig4)
    if 'risk_distribution' in analysis and len(analysis['risk_distribution']) > 0:
        fig5 = px.pie(values=analysis['risk_distribution'].values, names=analysis['risk_distribution'].index, title='Risk Level Distribution')
        fig5.update_layout(height=400)
        charts.append(fig5)
    return charts


# =============================
# 🧭 Streamlit App
# =============================

def main():
    st.set_page_config(page_title="🌍 QuakeGuard AI", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #1f77b4; margin-bottom: 2rem; }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-moderate { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #44aa44; font-weight: bold; }
    .risk-severe { color: #ff0000; font-weight: bold; }
    .risk-extreme { color: #800000; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; color: #222 !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🌍 QuakeGuardGPT</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time seismic monitoring with AI-powered risk assessment and emergency protocols")

    provider = get_active_llm_provider()
    with st.sidebar:
        st.header("⚙️ Configuration")
        if provider == "ai_ml":
            st.success("LLM Provider: AI/ML API (OpenAI-compatible)")
        elif provider == "groq":
            st.info("LLM Provider: Groq (fallback)")
        else:
            st.warning("LLM Provider: Not configured — AI analysis will be disabled")

    region = st.sidebar.text_input("🌍 Region (optional)", placeholder="e.g., California, Pakistan, Japan")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_magnitude = st.slider("📏 Min Magnitude", 1.0, 7.0, 2.5, 0.1)
    with col2:
        hours = st.slider("⏰ Hours", 1, 168, 24)
    with st.sidebar.expander("🔧 Advanced Options"):
        show_detailed_analysis = st.checkbox("Detailed Analysis", value=True)
        show_ai_summary = st.checkbox("AI Summary", value=True)
        show_emergency_protocols = st.checkbox("Emergency Protocols", value=True)

    region_bboxes = {
        "California": [-125, 32, -114, 42],
        "Pakistan": [60, 23, 77, 37],
        "Japan": [129, 31, 146, 45],
        "Chile": [-75, -56, -66, -17],
        "Turkey": [25, 36, 45, 43],
        "Indonesia": [95, -11, 141, 6],
        "India": [68, 6, 97, 37],
        "Mexico": [-118, 14, -86, 33],
        "USA": [-125, 24, -66, 49],
        "World": [-180, -90, 180, 90]
    }
    region_bbox = region_bboxes.get(region.strip().title()) if region else None
    if st.button("🔄 Refresh Data", type="primary"):
        st.rerun()

    with st.spinner("🌐 Fetching earthquake data..."):
        df = fetch_earthquakes(min_magnitude, hours, region_bbox, show_detailed_analysis)

    if df.empty:
        st.warning("⚠️ No recent earthquakes found matching your criteria.")
        st.info("💡 Try reducing the minimum magnitude or increasing the time range.")
        st.markdown(
            """
        <div class="metric-card">
            <h3>🚨 Current Risk Level: <span class="risk-low">LOW</span></h3>
            <p><strong>Risk Score:</strong> 0/80</p>
            <p><strong>Emergency Protocol:</strong> Monitor situation. No immediate action required.</p>
        </div>
            """, unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Map", "📊 Analytics", "📋 Data", "🤖 AI Analysis", "🚨 Emergency"]) 
        with tab1:
            st.subheader("🌍 Interactive Earthquake Map")
            st.info("No earthquake data available for map visualization")
        with tab2:
            st.subheader("📊 Advanced Analytics")
            st.info("No earthquake data available for analysis")
        with tab3:
            st.subheader("📋 Earthquake Data")
            st.info("No earthquake data available")
        with tab4:
            st.subheader("🤖 AI-Powered Analysis")
            if show_ai_summary:
                st.info("No earthquake data available for AI analysis")
            else:
                st.info("Enable AI Summary in Advanced Options to see AI analysis.")
        with tab5:
            st.subheader("🚨 Emergency Information")
            if show_emergency_protocols:
                st.markdown(
                    """
                    ### 🚨 Emergency Response Protocols
                    **Immediate Actions During Earthquake:**
                    - Drop, Cover, and Hold On
                    - Stay indoors if you're inside
                    - Move to open area if you're outside
                    - Stay away from windows, mirrors, and heavy objects
                    **After Earthquake:**
                    - Check for injuries and provide first aid
                    - Check for gas leaks and electrical damage
                    - Listen to emergency broadcasts
                    - Be prepared for aftershocks
                    **Emergency Contacts:**
                    - Emergency Services: 911 (US) / 112 (EU) / 999 (UK)
                    - USGS Earthquake Information: https://earthquake.usgs.gov
                    - Local Emergency Management: Check your local government website
                    """, unsafe_allow_html=True)
                st.markdown(
                    """
                    ### 📊 Current Emergency Status
                    - **Risk Level**: LOW
                    - **Recommended Action**: Monitor situation. No immediate action required.
                    - **Monitoring Required**: No
                    """, unsafe_allow_html=True)
            else:
                st.info("Enable Emergency Protocols in Advanced Options to see emergency information.")
        return

    # If we have data
    st.success(f"✅ Found {len(df)} earthquakes in the last {hours} hours")
    st.write(f"🕐 Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    risk_level, risk_score = calculate_overall_risk(df)
    st.markdown(f"""
        <div class="metric-card">
            <h3>🚨 Current Risk Level: <span class="risk-{risk_level}">{risk_level.upper()}</span></h3>
            <p><strong>Risk Score:</strong> {risk_score}</p>
            <p><strong>Emergency Protocol:</strong> {EMERGENCY_PROTOCOLS[risk_level]}</p>
        </div>
        """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Earthquakes", len(df))
    with col2:
        st.metric("Max Magnitude", f"{df['magnitude'].max():.1f}")
    with col3:
        st.metric("Avg Magnitude", f"{df['magnitude'].mean():.2f}")
    with col4:
        st.metric("Avg Depth", f"{df['depth'].mean():.1f} km")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Map", "📊 Analytics", "📋 Data", "🤖 AI Analysis", "🚨 Emergency"]) 

    with tab1:
        st.subheader("🌍 Interactive Earthquake Map")
        try:
            map_obj = create_advanced_map(df, region_bbox)
            if map_obj:
                st_folium(map_obj, width=800, height=500)
            else:
                st.info("Unable to create map visualization")
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            st.info("Try adjusting your search criteria")

    with tab2:
        st.subheader("📊 Advanced Analytics")
        try:
            analysis = analyze_seismic_patterns(df)
            charts = create_comprehensive_charts(df, analysis)
            for chart in charts:
                st.plotly_chart(chart, use_container_width=True)
            if analysis:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📈 Magnitude Statistics")
                    if 'magnitude_stats' in analysis:
                        stats_df = pd.DataFrame([analysis['magnitude_stats']]).T
                        stats_df.columns = ['Value']
                        st.dataframe(stats_df)
                    else:
                        st.info("Insufficient data for magnitude statistics")
                with c2:
                    st.subheader("📊 Risk Distribution")
                    if 'risk_distribution' in analysis and len(analysis['risk_distribution']) > 0:
                        risk_df = pd.DataFrame(analysis['risk_distribution'])
                        risk_df.columns = ['Count']
                        st.dataframe(risk_df)
                    else:
                        st.info("No risk distribution data available")
        except Exception as e:
            st.error(f"Error in analytics: {str(e)}")
            st.info("Try adjusting your search criteria or check your internet connection")

    with tab3:
        st.subheader("📋 Earthquake Data")
        colA, colB = st.columns(2)
        with colA:
            magnitude_filter = st.multiselect("Filter by Magnitude Category", options=df['magnitude_category'].unique(), default=list(df['magnitude_category'].unique()))
        with colB:
            risk_filter = st.multiselect("Filter by Risk Level", options=df['risk_level'].unique(), default=list(df['risk_level'].unique()))
        filtered_df = df[(df['magnitude_category'].isin(magnitude_filter)) & (df['risk_level'].isin(risk_filter))]
        st.dataframe(filtered_df[['time', 'place', 'magnitude', 'depth', 'risk_level', 'time_ago', 'url']], use_container_width=True)
        csv = filtered_df.to_csv(index=False)
        st.download_button(label="📥 Download CSV", data=csv, file_name=f"earthquakes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    with tab4:
        st.subheader("🤖 AI-Powered Analysis")
        if (get_active_llm_provider() is not None):
            with st.spinner("🤖 Generating AI analysis..."):
                analysis = analyze_seismic_patterns(df)
                risk_level_cur, risk_score_cur = calculate_overall_risk(df)
                prompt = (
                    f"As an expert seismologist and emergency response specialist, provide a comprehensive analysis of the following earthquake data:\n"
                    f"SUMMARY STATISTICS:\n"
                    f"- Total earthquakes: {len(df)}\n"
                    f"- Time period: {hours} hours\n"
                    f"- Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}\n"
                    f"- Average magnitude: {df['magnitude'].mean():.2f}\n"
                    f"- Risk level: {risk_level_cur.upper()}\n"
                    f"- Risk score: {risk_score_cur}\n"
                    f"EARTHQUAKE DATA:\n{df[['time', 'place', 'magnitude', 'depth']].head(20).to_string(index=False)}\n\n"
                    f"Please provide:\n1. Risk Assessment – Detailed evaluation of current seismic risk\n2. Pattern Analysis – Identification of any concerning patterns or trends\n3. Regional Impact – Specific implications for affected areas\n4. Safety Recommendations – Detailed safety advice for the public\n5. Emergency Preparedness – Specific actions people should take\n6. Monitoring Recommendations – What to watch for in coming hours/days\nBe thorough, specific, and actionable in your response."
                )
                summary = get_llm_summary(prompt)
                st.markdown(summary)
        else:
            st.info("Configure AI_ML_API_KEY or GROQ_API_KEY to enable AI analysis in this tab.")

    with tab5:
        st.subheader("🚨 Emergency Information")
        if show_emergency_protocols:
            st.markdown(
                f"""
                ### 🚨 Emergency Response Protocols
                **Immediate Actions During Earthquake:**
                - Drop, Cover, and Hold On
                - Stay indoors if you're inside
                - Move to open area if you're outside
                - Stay away from windows, mirrors, and heavy objects
                **After Earthquake:**
                - Check for injuries and provide first aid
                - Check for gas leaks and electrical damage
                - Listen to emergency broadcasts
                - Be prepared for aftershocks
                **Emergency Contacts:**
                - Emergency Services: 911 (US) / 112 (EU) / 999 (UK)
                - USGS Earthquake Information: https://earthquake.usgs.gov
                - Local Emergency Management: Check your local government website
                ### 📊 Current Emergency Status
                - **Risk Level**: {risk_level.upper()}
                - **Recommended Action**: {EMERGENCY_PROTOCOLS[risk_level]}
                - **Monitoring Required**: {'Yes' if risk_level in ['high', 'severe', 'extreme'] else 'No'}
                """, unsafe_allow_html=True)
        else:
            st.info("Enable Emergency Protocols in Advanced Options to see emergency information.")


if __name__ == "__main__":
    main()
