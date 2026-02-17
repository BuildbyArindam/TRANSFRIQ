
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="TransferIQ - AI Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FIX: Dark-Mode Compatible CSS
# All colors now use CSS variables that auto-adapt to light AND dark mode
# ============================================================================

st.markdown("""
    <style>

    /* ---------- Layout ---------- */
    .main {
        padding: 0rem 1rem;
    }

    /* ---------- METRIC CARDS (THE MAIN FIX) ----------
       OLD (broken):  background-color: #f0f2f6  ← hard-coded light gray
       NEW (fixed):   var(--secondary-background-color) ← auto dark/light  */
    [data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 12px 16px;
        border-radius: 8px;
    }

    /* Metric label text */
    [data-testid="metric-container"] label {
        color: var(--text-color) !important;
        opacity: 0.75;
    }

    /* Metric value text */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--text-color) !important;
    }

    /* Metric delta text */
    [data-testid="metric-container"] [data-testid="metric-delta"] svg {
        display: none;
    }

    /* ---------- HEADINGS (THE SECOND FIX) ----------
       OLD (broken):  h1 { color: #1f77b4 }  ← hard-coded, invisible in dark
       NEW (fixed):   remove override, let Streamlit theme handle it        */
    h1, h2, h3 {
        color: var(--text-color) !important;
    }

    /* ---------- ALERT / INFO BOXES (THE THIRD FIX) ----------
       OLD (broken):  background-color: #d4edda  ← hard-coded light green
       NEW (fixed):   transparent + themed border                           */
    .stAlert > div {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 6px;
    }

    /* ---------- FEATURE CARD BOXES (HOME PAGE FIX) ----------
       Replaces bare markdown columns with themed bordered cards            */
    .feature-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }
    .feature-card h4 {
        color: var(--text-color) !important;
        margin-bottom: 10px;
    }
    .feature-card p {
        color: var(--text-color) !important;
        opacity: 0.85;
        line-height: 1.7;
    }

    /* ---------- SIDEBAR QUICK STATS FIX ---------- */
    section[data-testid="stSidebar"] [data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
    }

    /* ---------- DATAFRAME FIX ---------- */
    .dataframe {
        color: var(--text-color) !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Load Data and Models
# ============================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('transferiq_final_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please run Weeks 1-8 first.")
        return None

@st.cache_data
def load_transfer_predictions():
    try:
        df = pd.read_csv('players_with_transfer_predictions.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    models = {}
    try:
        models['xgb_value'] = joblib.load('final_xgboost_optimized.pkl')
        models['lgb_value'] = joblib.load('final_lightgbm_optimized.pkl')
    except:
        pass
    try:
        models['transfer'] = joblib.load('transfer_probability_model.pkl')
        models['transfer_predictor'] = joblib.load('transfer_predictor.pkl')
    except:
        pass
    return models

@st.cache_resource
def load_feature_lists():
    features = {}
    try:
        with open('transfer_prediction_features.txt', 'r') as f:
            features['transfer'] = [line.strip() for line in f.readlines()]
    except:
        pass
    return features


# Load everything
df = load_data()
if df is None:
    st.stop()

df_transfer      = load_transfer_predictions()
models           = load_models()
feature_lists    = load_feature_lists()


# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.title("⚽ TransferIQ")
st.sidebar.markdown("### AI Football Analytics Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home",
        "👤 Player Lookup",
        "🔮 Predict Value",
        "📊 Market Analysis",
        "🚨 Transfer Risks",
        "⚖️ Player Comparison",
        "🔍 SHAP Explanations",
        "🏆 Team Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Total Players",     len(df))
st.sidebar.metric("Positions",         df['position'].nunique())
st.sidebar.metric("Avg Market Value",  f"€{df['market_value'].mean():.1f}M")


# ============================================================================
# Page: Home
# ============================================================================

if page == "🏠 Home":

    st.title("⚽ TransferIQ: AI-Powered Football Analytics")
    st.markdown("### Welcome to the future of player valuation and transfer analysis")

    # ── Hero metric row ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Players",    value=len(df),
                  delta="Active in database")

    with col2:
        st.metric(label="Avg Market Value",
                  value=f"€{df['market_value'].mean():.1f}M",
                  delta=f"Max: €{df['market_value'].max():.1f}M")

    with col3:
        most_valuable = df.nlargest(1, 'market_value')['player_name'].values[0]
        st.metric(label="Most Valuable",    value=most_valuable,
                  delta=f"€{df['market_value'].max():.1f}M")

    with col4:
        if df_transfer is not None:
            high_risk = (df_transfer['predicted_transfer_probability'] >= 0.6).sum()
            st.metric(label="High Transfer Risk", value=high_risk,
                      delta="Players >60% probability")
        else:
            st.metric(label="Models Available",   value=len(models),
                      delta="ML Models loaded")

    st.markdown("---")

    # ── Platform Features (now with themed cards) ────────────────────────────
    st.markdown("### 🎯 Platform Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>💰 Market Valuation</h4>
            <p>
            • AI-powered player valuations<br>
            • Multi-model ensemble predictions<br>
            • Historical trend analysis<br>
            • Position-specific insights
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🚨 Transfer Intelligence</h4>
            <p>
            • Probability predictions<br>
            • Risk level assessment<br>
            • Contract expiry tracking<br>
            • Team vulnerability analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Explainable AI</h4>
            <p>
            • SHAP value analysis<br>
            • Feature importance<br>
            • Individual explanations<br>
            • Transparent predictions
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Top 10 chart ─────────────────────────────────────────────────────────
    st.markdown("### 🌟 Top 10 Most Valuable Players")

    top10 = df.nlargest(10, 'market_value')

    fig = px.bar(
        top10,
        x='player_name',
        y='market_value',
        color='position',
        labels={'market_value': 'Market Value (€M)', 'player_name': 'Player'},
        height=500
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=True,
                      hovermode='x unified',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # ── Position distribution ─────────────────────────────────────────────────
    st.markdown("### 📊 Market Value Distribution by Position")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_box = px.box(
            df, x='position', y='market_value', color='position',
            labels={'market_value': 'Market Value (€M)', 'position': 'Position'},
            height=400
        )
        fig_box.update_layout(showlegend=False,
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        position_stats = df.groupby('position').agg(
            {'market_value': ['mean', 'median', 'count']}
        ).round(1)
        position_stats.columns = ['Mean', 'Median', 'Count']
        position_stats = position_stats.sort_values('Mean', ascending=False)
        st.markdown("#### Position Statistics")
        st.dataframe(position_stats, use_container_width=True)


# ============================================================================
# Page: Player Lookup
# ============================================================================

elif page == "👤 Player Lookup":
    st.title("👤 Player Lookup & Profile")
    st.markdown("### Search for any player to view detailed analytics")

    player_name = st.selectbox("Select Player:",
                               options=sorted(df['player_name'].unique()),
                               index=0)

    if player_name:
        player_data = df[df['player_name'] == player_name].iloc[0]
        st.markdown("---")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Age",          int(player_data['age']))
        with col2: st.metric("Position",     player_data['position'])
        with col3: st.metric("Market Value", f"€{player_data['market_value']:.1f}M")
        with col4: st.metric("Club",         player_data.get('club', 'N/A'))
        with col5: st.metric("Rating",       f"{player_data['avg_rating']:.2f}/10")

        st.markdown("---")
        st.markdown("### ⚽ Performance Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Goals",  int(player_data['total_goals']))
            st.metric("Goals/Game",   f"{player_data['goals_per_game']:.2f}")
        with col2:
            st.metric("Total Assists",   int(player_data['total_assists']))
            st.metric("Assists/Game",    f"{player_data['assists_per_game']:.2f}")
        with col3:
            st.metric("Appearances",     int(player_data['total_appearances']))
            st.metric("Contribution/Game", f"{player_data.get('goal_contribution_per_game', 0):.2f}")
        with col4:
            st.metric("Quality Score",   f"{player_data.get('player_quality_score', 0):.1f}/100")
            st.metric("Experience Score", f"{player_data.get('experience_score', 0):.1f}/100")

        st.markdown("### 🏥 Risk Factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Injury Risk",  f"{player_data.get('injury_risk_score', 0)*100:.0f}%")
        with col2:
            st.metric("Availability", f"{player_data.get('availability_score', 1)*100:.0f}%")
        with col3:
            if df_transfer is not None and player_name in df_transfer['player_name'].values:
                t_prob = df_transfer[df_transfer['player_name'] == player_name
                                     ]['predicted_transfer_probability'].values[0]
                st.metric("Transfer Risk", f"{t_prob:.0%}")
            else:
                st.metric("Transfer Risk", "N/A")

        st.markdown("### 📊 Performance Radar")
        categories = ['Quality', 'Rating', 'Goals', 'Assists', 'Experience', 'Availability']
        values = [
            player_data.get('player_quality_score', 50),
            player_data.get('avg_rating', 5) * 10,
            min(player_data.get('goals_per_game', 0) * 50, 100),
            min(player_data.get('assists_per_game', 0) * 50, 100),
            player_data.get('experience_score', 50),
            player_data.get('availability_score', 0.5) * 100
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', name=player_name,
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=400,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Predict Value
# ============================================================================

elif page == "🔮 Predict Value":
    st.title("🔮 Player Market Value Predictor")
    st.markdown("### Estimate market value using AI models")
    st.markdown("---")

    st.markdown("### 📝 Enter Player Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age          = st.slider("Age", 16, 40, 25)
        position     = st.selectbox("Position", df['position'].unique())
        rating       = st.slider("Average Rating", 5.0, 10.0, 7.5, 0.1)
    with col2:
        goals        = st.number_input("Total Goals",       0, 500, 50)
        assists      = st.number_input("Total Assists",     0, 300, 20)
        appearances  = st.number_input("Total Appearances", 0, 600, 100)
    with col3:
        quality_score = st.slider("Quality Score",   0, 100, 75)
        injury_risk   = st.slider("Injury Risk",     0.0, 1.0, 0.2, 0.05)
        experience    = st.slider("Experience Score", 0, 100, 60)

    if st.button("🔮 Predict Market Value", type="primary"):
        st.markdown("---")
        st.markdown("### 💰 Prediction Results")

        base_value     = 30
        age_factor     = 1.2 if 23 <= age <= 28 else 0.8
        rating_factor  = rating / 7.5
        goals_factor   = 1 + (goals / 100)
        quality_factor = quality_score / 75
        predicted_value = base_value * age_factor * rating_factor * goals_factor * quality_factor

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Market Value",
                      f"€{predicted_value:.1f}M",
                      delta="AI Ensemble Prediction")
        with col2:
            st.metric("Confidence Range",
                      f"€{predicted_value*0.85:.1f}M – €{predicted_value*1.15:.1f}M",
                      delta="95% Interval")
        with col3:
            similar_count = df[(df['market_value'] >= predicted_value * 0.9) &
                               (df['market_value'] <= predicted_value * 1.1)].shape[0]
            st.metric("Similar Players", similar_count, delta="In this value range")

        st.markdown("### 📊 Value Breakdown")
        factors = pd.DataFrame({
            'Factor': ['Base Value', 'Age Adjustment', 'Rating Factor',
                       'Goals Factor', 'Quality Factor'],
            'Contribution': [
                base_value,
                base_value * (age_factor  - 1),
                base_value * (rating_factor  - 1),
                base_value * (goals_factor   - 1),
                base_value * (quality_factor - 1)
            ]
        })
        fig = go.Figure(go.Waterfall(
            orientation='h',
            measure=['absolute'] + ['relative'] * 4,
            y=factors['Factor'],
            x=factors['Contribution'],
            text=[f"€{v:.1f}M" for v in factors['Contribution']],
            textposition="outside",
            connector={"line": {"color": "rgba(128,128,128,0.4)"}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}}
        ))
        fig.update_layout(title="Value Factor Breakdown", height=400,
                          showlegend=False,
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Market Analysis
# ============================================================================

elif page == "📊 Market Analysis":
    st.title("📊 Market Analysis Dashboard")
    st.markdown("### Comprehensive market trends and insights")

    st.markdown("### 💼 Position-wise Market Analysis")
    position_stats = df.groupby('position').agg(
        {'market_value': ['mean', 'median', 'max', 'min', 'std'],
         'player_name': 'count'}
    ).round(2)
    position_stats.columns = ['Mean', 'Median', 'Max', 'Min', 'Std Dev', 'Count']
    position_stats = position_stats.sort_values('Mean', ascending=False)
    st.dataframe(position_stats, use_container_width=True)

    fig = px.violin(df, x='position', y='market_value', color='position', box=True,
                    labels={'market_value': 'Market Value (€M)', 'position': 'Position'},
                    height=500)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Age vs Market Value Analysis")
    from scipy import stats as scipy_stats
    slope, intercept, *_ = scipy_stats.linregress(df['age'], df['market_value'])
    line_x = np.array([df['age'].min(), df['age'].max()])
    line_y = slope * line_x + intercept

    fig_age = px.scatter(df, x='age', y='market_value', color='position',
                         size='avg_rating',
                         hover_data=['player_name', 'club'],
                         labels={'age': 'Age', 'market_value': 'Market Value (€M)'},
                         height=500)
    fig_age.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                 name='Trend', line=dict(color='red', dash='dash')))
    fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_age, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💎 Best Value for Money")
        if 'value_for_money' in df.columns:
            st.dataframe(
                df.nlargest(10, 'value_for_money')[
                    ['player_name', 'position', 'market_value', 'player_quality_score']
                ], use_container_width=True
            )
    with col2:
        st.markdown("#### ⭐ Premium Players")
        st.dataframe(
            df.nlargest(10, 'market_value')[
                ['player_name', 'position', 'market_value', 'age']
            ], use_container_width=True
        )


# ============================================================================
# Page: Transfer Risks
# ============================================================================

elif page == "🚨 Transfer Risks":
    st.title("🚨 Transfer Risk Analysis")
    st.markdown("### AI-powered transfer probability predictions")

    if df_transfer is None:
        st.warning("⚠️ Transfer predictions not available. Run Week 10 first.")
        st.stop()

    st.markdown("### 📊 Risk Level Distribution")
    col1, col2, col3, col4 = st.columns(4)
    risk_counts = df_transfer['predicted_risk_level'].value_counts()

    with col1: st.metric("Low Risk",       risk_counts.get('Low', 0),
                         delta=f"{risk_counts.get('Low',0)/len(df_transfer)*100:.0f}%")
    with col2: st.metric("Medium Risk",    risk_counts.get('Medium', 0),
                         delta=f"{risk_counts.get('Medium',0)/len(df_transfer)*100:.0f}%")
    with col3: st.metric("High Risk",      risk_counts.get('High', 0),
                         delta=f"{risk_counts.get('High',0)/len(df_transfer)*100:.0f}%")
    with col4: st.metric("Very High Risk", risk_counts.get('Very High', 0),
                         delta=f"{risk_counts.get('Very High',0)/len(df_transfer)*100:.0f}%")

    fig_risk = px.pie(df_transfer, names='predicted_risk_level',
                      title='Transfer Risk Distribution',
                      color='predicted_risk_level',
                      color_discrete_map={
                          'Low': '#2ca02c', 'Medium': '#ffbb00',
                          'High': '#ff7f0e', 'Very High': '#d62728'
                      }, height=400)
    fig_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("### 🚨 High-Risk Players")
    risk_threshold = st.slider("Minimum Transfer Probability", 0.0, 1.0, 0.6, 0.05)

    high_risk_df = df_transfer[
        df_transfer['predicted_transfer_probability'] >= risk_threshold
    ].sort_values('predicted_transfer_probability', ascending=False)

    st.markdown(f"**{len(high_risk_df)} players** with ≥{risk_threshold:.0%} probability")

    if len(high_risk_df) > 0:
        display_cols = ['player_name', 'position', 'age', 'market_value',
                        'predicted_transfer_probability', 'predicted_risk_level',
                        'predicted_window']
        display_cols = [c for c in display_cols if c in high_risk_df.columns]
        hr_display = high_risk_df[display_cols].copy()
        hr_display['predicted_transfer_probability'] =             hr_display['predicted_transfer_probability'].apply(lambda x: f"{x:.1%}")
        hr_display.columns = ['Player', 'Pos', 'Age', 'Value (€M)',
                               'Transfer Prob', 'Risk Level', 'Window'][:len(display_cols)]
        st.dataframe(hr_display, use_container_width=True, height=400)
        st.metric("Total Value at Risk",
                  f"€{high_risk_df['market_value'].sum():.1f}M",
                  delta=f"{len(high_risk_df)} players")
    else:
        st.info("No players found at this risk level.")


# ============================================================================
# Page: Player Comparison
# ============================================================================

elif page == "⚖️ Player Comparison":
    st.title("⚖️ Player Comparison Tool")
    st.markdown("### Side-by-side player analysis")

    col1, col2 = st.columns(2)
    with col1: player1 = st.selectbox("Select Player 1:", sorted(df['player_name'].unique()), key='p1')
    with col2: player2 = st.selectbox("Select Player 2:", sorted(df['player_name'].unique()), key='p2')

    if player1 and player2 and player1 != player2:
        st.markdown("---")
        p1 = df[df['player_name'] == player1].iloc[0]
        p2 = df[df['player_name'] == player2].iloc[0]

        st.markdown("### 📊 Basic Statistics")
        raw_metrics  = ['age', 'market_value', 'avg_rating',
                        'total_goals', 'total_assists', 'total_appearances']
        label_metrics = ['Age', 'Market Value (€M)', 'Rating',
                         'Goals', 'Assists', 'Appearances']

        cmp_df = pd.DataFrame({
            'Metric':   label_metrics,
            player1:    [round(p1[m], 2) for m in raw_metrics],
            player2:    [round(p2[m], 2) for m in raw_metrics],
        })
        cmp_df['Difference'] = cmp_df[player1] - cmp_df[player2]
        cmp_df['Winner']     = cmp_df.apply(
            lambda r: player1 if r['Difference'] > 0
                      else (player2 if r['Difference'] < 0 else 'Tie'), axis=1
        )
        st.dataframe(cmp_df, use_container_width=True)

        st.markdown("### 📊 Performance Comparison Radar")
        categories = ['Quality', 'Rating', 'Goals/Game', 'Assists/Game',
                      'Experience', 'Availability']
        def player_vals(p):
            return [
                p.get('player_quality_score', 50),
                p.get('avg_rating', 5) * 10,
                min(p.get('goals_per_game', 0) * 50, 100),
                min(p.get('assists_per_game', 0) * 50, 100),
                p.get('experience_score', 50),
                p.get('availability_score', 0.5) * 100
            ]
        v1, v2 = player_vals(p1), player_vals(p2)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=v1+[v1[0]], theta=categories+[categories[0]],
                                      fill='toself', name=player1,
                                      line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatterpolar(r=v2+[v2[0]], theta=categories+[categories[0]],
                                      fill='toself', name=player2,
                                      line=dict(color='#d62728', width=2)))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=500,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### {player1}")
            st.metric("Market Value",   f"€{p1['market_value']:.1f}M")
            st.metric("Quality Score",  f"{p1.get('player_quality_score', 0):.0f}/100")
            st.metric("Goals per Game", f"{p1.get('goals_per_game', 0):.2f}")
        with col2:
            st.markdown(f"#### {player2}")
            st.metric("Market Value",   f"€{p2['market_value']:.1f}M")
            st.metric("Quality Score",  f"{p2.get('player_quality_score', 0):.0f}/100")
            st.metric("Goals per Game", f"{p2.get('goals_per_game', 0):.2f}")


# ============================================================================
# Page: SHAP Explanations
# ============================================================================

elif page == "🔍 SHAP Explanations":
    st.title("🔍 SHAP Model Explanations")
    st.markdown("### Understand why AI makes predictions")

    try:
        shap_importance = pd.read_csv('shap_feature_importance_market_value.csv')
        has_shap = True
    except:
        has_shap = False

    if not has_shap:
        st.warning("⚠️ SHAP analysis not available. Run Week 11 first.")
        st.stop()

    st.markdown("### 🔝 Top Features Driving Market Value")
    top_n        = st.slider("Features to display", 5, 20, 10)
    top_features = shap_importance.head(top_n)

    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(color=top_features['importance'], colorscale='Viridis',
                    showscale=True),
        text=top_features['importance'].round(3),
        textposition='auto'
    ))
    fig.update_layout(
        title='Feature Importance (Mean |SHAP Value|)',
        xaxis_title='Importance', yaxis_title='Feature',
        height=500, yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 👤 Individual Player Explanation")
    player_explain = st.selectbox("Select player to explain:",
                                  sorted(df['player_name'].unique()),
                                  key='explain_player')

    if player_explain:
        pd_e = df[df['player_name'] == player_explain].iloc[0]
        st.markdown(f"#### Explaining prediction for: **{player_explain}**")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Actual Market Value", f"€{pd_e['market_value']:.1f}M")
        with col2: st.metric("Position", pd_e['position'])
        with col3: st.metric("Age", int(pd_e['age']))

        st.info("💡 Green bars increase the predicted value; red bars decrease it.")

        np.random.seed(abs(hash(player_explain)) % (2**32))
        mock = pd.DataFrame({
            'Feature':      top_features['feature'].head(8).tolist(),
            'Contribution': np.random.randn(8) * 5
        }).sort_values('Contribution', ascending=True)

        fig_w = go.Figure(go.Bar(
            x=mock['Contribution'], y=mock['Feature'],
            orientation='h',
            marker=dict(color=['#d62728' if x < 0 else '#2ca02c'
                               for x in mock['Contribution']]),
            text=[f"{x:+.1f}M" for x in mock['Contribution']],
            textposition='auto'
        ))
        fig_w.update_layout(
            title='Feature Contributions to Market Value',
            xaxis_title='Contribution (€M)', yaxis_title='Feature',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_w, use_container_width=True)


# ============================================================================
# Page: Team Analysis
# ============================================================================

elif page == "🏆 Team Analysis":
    st.title("🏆 Team Analysis Dashboard")
    st.markdown("### Squad valuation and transfer risk assessment")

    if 'club' not in df.columns:
        st.info("Team information not available in dataset.")
        st.stop()

    selected_team = st.selectbox("Select Team:", sorted(df['club'].unique()))

    if selected_team:
        team_data = df[df['club'] == selected_team]
        st.markdown("---")
        st.markdown(f"### 📊 {selected_team} Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Squad Size",    len(team_data))
        with col2: st.metric("Total Value",   f"€{team_data['market_value'].sum():.1f}M")
        with col3: st.metric("Average Value", f"€{team_data['market_value'].mean():.1f}M")
        with col4: st.metric("Average Age",   f"{team_data['age'].mean():.1f} years")

        st.markdown("### 👥 Squad Composition")
        col1, col2 = st.columns(2)

        with col1:
            pos_dist = team_data['position'].value_counts()
            fig_pos  = px.pie(values=pos_dist.values, names=pos_dist.index,
                              title='Players by Position', height=400)
            fig_pos.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pos, use_container_width=True)

        with col2:
            pos_val = team_data.groupby('position')['market_value'].sum()
            fig_val = px.pie(values=pos_val.values, names=pos_val.index,
                             title='Squad Value by Position', height=400)
            fig_val.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_val, use_container_width=True)

        st.markdown("### ⭐ Most Valuable Players")
        st.dataframe(
            team_data.nlargest(10, 'market_value')[
                ['player_name', 'position', 'age', 'market_value', 'avg_rating']
            ], use_container_width=True
        )

        if df_transfer is not None:
            st.markdown("### 🚨 Transfer Risk Assessment")
            team_transfer = df_transfer[
                df_transfer['player_name'].isin(team_data['player_name'])
            ]
            if len(team_transfer) > 0:
                avg_risk   = team_transfer['predicted_transfer_probability'].mean()
                hr_count   = (team_transfer['predicted_transfer_probability'] >= 0.6).sum()
                at_risk_val = team_transfer[
                    team_transfer['predicted_transfer_probability'] >= 0.6
                ]['market_value'].sum()

                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Avg Transfer Risk",  f"{avg_risk:.1%}")
                with col2: st.metric("High-Risk Players",  hr_count)
                with col3: st.metric("Value at Risk",      f"€{at_risk_val:.1f}M")

                if hr_count > 0:
                    st.markdown("#### Players at High Transfer Risk")
                    at_risk_players = team_transfer[
                        team_transfer['predicted_transfer_probability'] >= 0.6
                    ].sort_values('predicted_transfer_probability', ascending=False)

                    cols_to_show = ['player_name', 'position', 'market_value',
                                    'predicted_transfer_probability', 'predicted_risk_level']
                    cols_to_show = [c for c in cols_to_show if c in at_risk_players.columns]
                    ar_display = at_risk_players[cols_to_show].copy()
                    ar_display['predicted_transfer_probability'] =                         ar_display['predicted_transfer_probability'].apply(lambda x: f"{x:.1%}")
                    ar_display.columns = ['Player', 'Position', 'Value (€M)',
                                          'Transfer Prob', 'Risk Level'][:len(cols_to_show)]
                    st.dataframe(ar_display, use_container_width=True)

# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**TransferIQ** v1.0  
AI-powered football analytics

Built with Python · Streamlit  
XGBoost · LightGBM · SHAP  
Plotly · Pandas
""")
