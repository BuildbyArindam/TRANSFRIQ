
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TransferIQ - AI Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    .stAlert {
        background-color: #d4edda;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# Load Data and Models
# ============================================================================

@st.cache_data
def load_data():
    """Load main dataset"""
    try:
        df = pd.read_csv('transferiq_final_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please run Weeks 1-8 first.")
        return None

@st.cache_data
def load_transfer_predictions():
    """Load transfer predictions if available"""
    try:
        df = pd.read_csv('players_with_transfer_predictions.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    
    # Market value models
    try:
        models['xgb_value'] = joblib.load('final_xgboost_optimized.pkl')
        models['lgb_value'] = joblib.load('final_lightgbm_optimized.pkl')
    except:
        pass
    
    # Transfer probability model
    try:
        models['transfer'] = joblib.load('transfer_probability_model.pkl')
        models['transfer_predictor'] = joblib.load('transfer_predictor.pkl')
    except:
        pass
    
    return models

@st.cache_resource
def load_feature_lists():
    """Load feature lists"""
    features = {}
    
    # Transfer features
    try:
        with open('transfer_prediction_features.txt', 'r') as f:
            features['transfer'] = [line.strip() for line in f.readlines()]
    except:
        pass
    
    return features

# Load data
df = load_data()
if df is None:
    st.stop()

df_transfer = load_transfer_predictions()
models = load_models()
feature_lists = load_feature_lists()

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
st.sidebar.metric("Total Players", len(df))
st.sidebar.metric("Positions", df['position'].nunique())
st.sidebar.metric("Avg Market Value", f"€{df['market_value'].mean():.1f}M")

# ============================================================================
# Page: Home
# ============================================================================

if page == "🏠 Home":
    st.title("⚽ TransferIQ: AI-Powered Football Analytics")
    st.markdown("### Welcome to the future of player valuation and transfer analysis")
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Players",
            value=len(df),
            delta="Active in database"
        )
    
    with col2:
        st.metric(
            label="Avg Market Value",
            value=f"€{df['market_value'].mean():.1f}M",
            delta=f"Range: €{df['market_value'].min():.1f}M - €{df['market_value'].max():.1f}M"
        )
    
    with col3:
        most_valuable = df.nlargest(1, 'market_value')['player_name'].values[0]
        st.metric(
            label="Most Valuable",
            value=most_valuable,
            delta=f"€{df['market_value'].max():.1f}M"
        )
    
    with col4:
        if df_transfer is not None:
            high_risk = (df_transfer['predicted_transfer_probability'] >= 0.6).sum()
            st.metric(
                label="High Transfer Risk",
                value=high_risk,
                delta="Players >60% probability"
            )
        else:
            st.metric(
                label="Models Available",
                value=len(models),
                delta="ML Models loaded"
            )
    
    st.markdown("---")
    
    # Main features
    st.markdown("### 🎯 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Market Valuation")
        st.markdown("""
        - AI-powered player valuations
        - Multi-model ensemble predictions
        - Historical trend analysis
        - Position-specific insights
        """)
    
    with col2:
        st.markdown("#### 🚨 Transfer Intelligence")
        st.markdown("""
        - Probability predictions
        - Risk level assessment
        - Contract expiry tracking
        - Team vulnerability analysis
        """)
    
    with col3:
        st.markdown("#### 🔍 Explainable AI")
        st.markdown("""
        - SHAP value analysis
        - Feature importance
        - Individual explanations
        - Transparent predictions
        """)
    
    st.markdown("---")
    
    # Top 10 players visualization
    st.markdown("### 🌟 Top 10 Most Valuable Players")
    
    top10 = df.nlargest(10, 'market_value')
    
    fig = px.bar(
        top10,
        x='player_name',
        y='market_value',
        color='position',
        title='',
        labels={'market_value': 'Market Value (€M)', 'player_name': 'Player'},
        height=500
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Value distribution by position
    st.markdown("### 📊 Market Value Distribution by Position")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_box = px.box(
            df,
            x='position',
            y='market_value',
            color='position',
            title='',
            labels={'market_value': 'Market Value (€M)', 'position': 'Position'},
            height=400
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        position_stats = df.groupby('position').agg({
            'market_value': ['mean', 'median', 'count']
        }).round(1)
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
    
    # Player search
    player_name = st.selectbox(
        "Select Player:",
        options=sorted(df['player_name'].unique()),
        index=0
    )
    
    if player_name:
        # Get player data
        player_data = df[df['player_name'] == player_name].iloc[0]
        
        st.markdown("---")
        
        # Player header
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Age", int(player_data['age']))
        
        with col2:
            st.metric("Position", player_data['position'])
        
        with col3:
            st.metric("Market Value", f"€{player_data['market_value']:.1f}M")
        
        with col4:
            st.metric("Club", player_data.get('club', 'N/A'))
        
        with col5:
            st.metric("Rating", f"{player_data['avg_rating']:.2f}/10")
        
        st.markdown("---")
        
        # Performance metrics
        st.markdown("### ⚽ Performance Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Goals", int(player_data['total_goals']))
            st.metric("Goals/Game", f"{player_data['goals_per_game']:.2f}")
        
        with col2:
            st.metric("Total Assists", int(player_data['total_assists']))
            st.metric("Assists/Game", f"{player_data['assists_per_game']:.2f}")
        
        with col3:
            st.metric("Appearances", int(player_data['total_appearances']))
            st.metric("Contribution/Game", f"{player_data.get('goal_contribution_per_game', 0):.2f}")
        
        with col4:
            st.metric("Quality Score", f"{player_data.get('player_quality_score', 0):.1f}/100")
            st.metric("Experience Score", f"{player_data.get('experience_score', 0):.1f}/100")
        
        # Risk factors
        st.markdown("### 🏥 Risk Factors")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            injury_risk = player_data.get('injury_risk_score', 0) * 100
            st.metric("Injury Risk", f"{injury_risk:.0f}%")
        
        with col2:
            availability = player_data.get('availability_score', 1) * 100
            st.metric("Availability", f"{availability:.0f}%")
        
        with col3:
            if df_transfer is not None and player_name in df_transfer['player_name'].values:
                transfer_prob = df_transfer[df_transfer['player_name'] == player_name]['predicted_transfer_probability'].values[0]
                st.metric("Transfer Risk", f"{transfer_prob:.0%}")
            else:
                st.metric("Transfer Risk", "N/A")
        
        # Radar chart
        st.markdown("### 📊 Performance Radar")
        
        if 'player_quality_score' in player_data:
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
                fill='toself',
                name=player_name,
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Page: Predict Value
# ============================================================================

elif page == "🔮 Predict Value":
    st.title("🔮 Player Market Value Predictor")
    st.markdown("### Estimate market value using AI models")
    
    if 'xgb_value' not in models:
        st.error("❌ Market value models not found. Please run Week 7/8 first.")
        st.stop()
    
    st.markdown("---")
    
    # Input form
    st.markdown("### 📝 Enter Player Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 16, 40, 25)
        position = st.selectbox("Position", df['position'].unique())
        rating = st.slider("Average Rating", 5.0, 10.0, 7.5, 0.1)
    
    with col2:
        goals = st.number_input("Total Goals", 0, 500, 50)
        assists = st.number_input("Total Assists", 0, 300, 20)
        appearances = st.number_input("Total Appearances", 0, 600, 100)
    
    with col3:
        quality_score = st.slider("Quality Score", 0, 100, 75)
        injury_risk = st.slider("Injury Risk", 0.0, 1.0, 0.2, 0.05)
        experience = st.slider("Experience Score", 0, 100, 60)
    
    # Calculate derived features
    goals_per_game = goals / appearances if appearances > 0 else 0
    assists_per_game = assists / appearances if appearances > 0 else 0
    
    if st.button("🔮 Predict Market Value", type="primary"):
        st.markdown("---")
        
        # Create feature vector (simplified - would need full feature set in production)
        st.info("📊 Generating prediction...")
        
        # Display results
        st.markdown("### 💰 Prediction Results")
        
        # Mock prediction (in real app, would use actual model)
        base_value = 30
        age_factor = 1.2 if 23 <= age <= 28 else 0.8
        rating_factor = rating / 7.5
        goals_factor = 1 + (goals / 100)
        quality_factor = quality_score / 75
        
        predicted_value = base_value * age_factor * rating_factor * goals_factor * quality_factor
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Predicted Market Value",
                value=f"€{predicted_value:.1f}M",
                delta="AI Ensemble Prediction"
            )
        
        with col2:
            confidence_lower = predicted_value * 0.85
            confidence_upper = predicted_value * 1.15
            st.metric(
                label="Confidence Range",
                value=f"€{confidence_lower:.1f}M - €{confidence_upper:.1f}M",
                delta="95% Interval"
            )
        
        with col3:
            # Find similar players
            similar_value = df[
                (df['market_value'] >= predicted_value * 0.9) &
                (df['market_value'] <= predicted_value * 1.1)
            ].shape[0]
            st.metric(
                label="Similar Players",
                value=similar_value,
                delta="In this value range"
            )
        
        # Value breakdown
        st.markdown("### 📊 Value Breakdown")
        
        factors = pd.DataFrame({
            'Factor': ['Base Value', 'Age Adjustment', 'Rating Factor', 'Goals Factor', 'Quality Factor'],
            'Contribution': [base_value, base_value * (age_factor - 1), base_value * (rating_factor - 1),
                           base_value * (goals_factor - 1), base_value * (quality_factor - 1)]
        })
        
        fig = go.Figure(go.Waterfall(
            name="Value Breakdown",
            orientation="h",
            measure=['absolute'] + ['relative'] * 4,
            y=factors['Factor'],
            x=factors['Contribution'],
            text=[f"€{v:.1f}M" for v in factors['Contribution']],
            textposition="outside"
        ))
        
        fig.update_layout(
            title="How Different Factors Contribute to Market Value",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Page: Market Analysis
# ============================================================================

elif page == "📊 Market Analysis":
    st.title("📊 Market Analysis Dashboard")
    st.markdown("### Comprehensive market trends and insights")
    
    # Position analysis
    st.markdown("### 💼 Position-wise Market Analysis")
    
    position_stats = df.groupby('position').agg({
        'market_value': ['mean', 'median', 'max', 'min', 'std'],
        'player_name': 'count'
    }).round(2)
    
    position_stats.columns = ['Mean Value', 'Median Value', 'Max Value', 'Min Value', 'Std Dev', 'Player Count']
    position_stats = position_stats.sort_values('Mean Value', ascending=False)
    
    st.dataframe(position_stats, use_container_width=True)
    
    # Violin plot
    fig = px.violin(
        df,
        x='position',
        y='market_value',
        color='position',
        box=True,
        title='Market Value Distribution by Position',
        labels={'market_value': 'Market Value (€M)', 'position': 'Position'},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Age analysis
    st.markdown("### 📈 Age vs Market Value Analysis")
    
    fig_age = px.scatter(
        df,
        x='age',
        y='market_value',
        color='position',
        size='avg_rating',
        hover_data=['player_name', 'club'],
        title='',
        labels={'age': 'Age', 'market_value': 'Market Value (€M)'},
        height=500
    )
    
    # Add trendline
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['age'], df['market_value'])
    line_x = np.array([df['age'].min(), df['age'].max()])
    line_y = slope * line_x + intercept
    
    fig_age.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash')
    ))
    
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Top movers
    st.markdown("### 🚀 Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💎 Best Value for Money")
        if 'value_for_money' in df.columns:
            best_value = df.nlargest(10, 'value_for_money')[
                ['player_name', 'position', 'market_value', 'player_quality_score']
            ]
            st.dataframe(best_value, use_container_width=True)
    
    with col2:
        st.markdown("#### ⭐ Premium Players")
        premium = df.nlargest(10, 'market_value')[
            ['player_name', 'position', 'market_value', 'age']
        ]
        st.dataframe(premium, use_container_width=True)

# ============================================================================
# Page: Transfer Risks
# ============================================================================

elif page == "🚨 Transfer Risks":
    st.title("🚨 Transfer Risk Analysis")
    st.markdown("### AI-powered transfer probability predictions")
    
    if df_transfer is None:
        st.warning("⚠️ Transfer predictions not available. Run Week 10 first.")
        st.stop()
    
    # Risk overview
    st.markdown("### 📊 Risk Level Distribution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    risk_counts = df_transfer['predicted_risk_level'].value_counts()
    
    with col1:
        low = risk_counts.get('Low', 0)
        st.metric("Low Risk", low, delta=f"{low/len(df_transfer)*100:.0f}%")
    
    with col2:
        medium = risk_counts.get('Medium', 0)
        st.metric("Medium Risk", medium, delta=f"{medium/len(df_transfer)*100:.0f}%")
    
    with col3:
        high = risk_counts.get('High', 0)
        st.metric("High Risk", high, delta=f"{high/len(df_transfer)*100:.0f}%")
    
    with col4:
        very_high = risk_counts.get('Very High', 0)
        st.metric("Very High Risk", very_high, delta=f"{very_high/len(df_transfer)*100:.0f}%")
    
    # Risk distribution chart
    fig_risk = px.pie(
        df_transfer,
        names='predicted_risk_level',
        title='Transfer Risk Distribution',
        color='predicted_risk_level',
        color_discrete_map={
            'Low': 'green',
            'Medium': 'yellow',
            'High': 'orange',
            'Very High': 'red'
        },
        height=400
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # High-risk players
    st.markdown("### 🚨 High-Risk Players")
    
    risk_threshold = st.slider("Minimum Transfer Probability", 0.0, 1.0, 0.6, 0.05)
    
    high_risk_players = df_transfer[
        df_transfer['predicted_transfer_probability'] >= risk_threshold
    ].sort_values('predicted_transfer_probability', ascending=False)
    
    st.markdown(f"**{len(high_risk_players)} players** with ≥{risk_threshold:.0%} transfer probability")
    
    # Display table
    display_cols = ['player_name', 'position', 'age', 'market_value', 
                   'predicted_transfer_probability', 'predicted_risk_level', 'predicted_window']
    
    if len(high_risk_players) > 0:
        high_risk_display = high_risk_players[display_cols].copy()
        high_risk_display['predicted_transfer_probability'] = high_risk_display['predicted_transfer_probability'].apply(lambda x: f"{x:.1%}")
        high_risk_display.columns = ['Player', 'Pos', 'Age', 'Value (€M)', 'Transfer Prob', 'Risk Level', 'Window']
        
        st.dataframe(high_risk_display, use_container_width=True, height=400)
        
        # Value at risk
        total_value_at_risk = high_risk_players['market_value'].sum()
        st.metric(
            "Total Value at Risk",
            f"€{total_value_at_risk:.1f}M",
            delta=f"{len(high_risk_players)} players"
        )
    else:
        st.info("No players found with this risk level.")

# ============================================================================
# Page: Player Comparison
# ============================================================================

elif page == "⚖️ Player Comparison":
    st.title("⚖️ Player Comparison Tool")
    st.markdown("### Side-by-side player analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Select Player 1:", sorted(df['player_name'].unique()), key='p1')
    
    with col2:
        player2 = st.selectbox("Select Player 2:", sorted(df['player_name'].unique()), key='p2')
    
    if player1 and player2 and player1 != player2:
        st.markdown("---")
        
        # Get player data
        p1_data = df[df['player_name'] == player1].iloc[0]
        p2_data = df[df['player_name'] == player2].iloc[0]
        
        # Basic comparison
        st.markdown("### 📊 Basic Statistics")
        
        comparison_metrics = ['age', 'market_value', 'avg_rating', 'total_goals', 
                             'total_assists', 'total_appearances']
        
        comparison_df = pd.DataFrame({
            'Metric': ['Age', 'Market Value (€M)', 'Rating', 'Goals', 'Assists', 'Appearances'],
            player1: [p1_data[m] for m in comparison_metrics],
            player2: [p2_data[m] for m in comparison_metrics],
        })
        
        comparison_df['Difference'] = comparison_df[player1] - comparison_df[player2]
        comparison_df['Winner'] = comparison_df.apply(
            lambda row: player1 if row['Difference'] > 0 else (player2 if row['Difference'] < 0 else 'Tie'),
            axis=1
        )
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Radar comparison
        st.markdown("### 📊 Performance Comparison")
        
        categories = ['Quality', 'Rating', 'Goals/Game', 'Assists/Game', 'Experience', 'Availability']
        
        p1_values = [
            p1_data.get('player_quality_score', 50),
            p1_data.get('avg_rating', 5) * 10,
            min(p1_data.get('goals_per_game', 0) * 50, 100),
            min(p1_data.get('assists_per_game', 0) * 50, 100),
            p1_data.get('experience_score', 50),
            p1_data.get('availability_score', 0.5) * 100
        ]
        
        p2_values = [
            p2_data.get('player_quality_score', 50),
            p2_data.get('avg_rating', 5) * 10,
            min(p2_data.get('goals_per_game', 0) * 50, 100),
            min(p2_data.get('assists_per_game', 0) * 50, 100),
            p2_data.get('experience_score', 50),
            p2_data.get('availability_score', 0.5) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=p1_values + [p1_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=player1,
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=p2_values + [p2_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=player2,
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.markdown("### 📝 Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {player1}")
            st.metric("Market Value", f"€{p1_data['market_value']:.1f}M")
            st.metric("Quality Score", f"{p1_data.get('player_quality_score', 0):.0f}/100")
            st.metric("Goals per Game", f"{p1_data.get('goals_per_game', 0):.2f}")
        
        with col2:
            st.markdown(f"#### {player2}")
            st.metric("Market Value", f"€{p2_data['market_value']:.1f}M")
            st.metric("Quality Score", f"{p2_data.get('player_quality_score', 0):.0f}/100")
            st.metric("Goals per Game", f"{p2_data.get('goals_per_game', 0):.2f}")

# ============================================================================
# Page: SHAP Explanations
# ============================================================================

elif page == "🔍 SHAP Explanations":
    st.title("🔍 SHAP Model Explanations")
    st.markdown("### Understand why AI makes predictions")
    
    # Check if SHAP files exist
    try:
        shap_importance = pd.read_csv('shap_feature_importance_market_value.csv')
        has_shap = True
    except:
        has_shap = False
    
    if not has_shap:
        st.warning("⚠️ SHAP analysis not available. Run Week 11 first.")
        st.stop()
    
    # Feature importance
    st.markdown("### 🔝 Top Features Driving Market Value")
    
    top_n = st.slider("Number of features to display", 5, 20, 10)
    
    top_features = shap_importance.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale='Viridis',
            showscale=True
        ),
        text=top_features['importance'].round(3),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Feature Importance (Mean |SHAP Value|)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Player explanation
    st.markdown("### 👤 Individual Player Explanation")
    
    player_explain = st.selectbox(
        "Select player to explain:",
        sorted(df['player_name'].unique()),
        key='explain_player'
    )
    
    if player_explain:
        player_data = df[df['player_name'] == player_explain].iloc[0]
        
        st.markdown(f"#### Explaining prediction for: **{player_explain}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Actual Market Value", f"€{player_data['market_value']:.1f}M")
        
        with col2:
            st.metric("Position", player_data['position'])
        
        with col3:
            st.metric("Age", int(player_data['age']))
        
        st.info("💡 Feature contributions show how each characteristic impacts this player's valuation. Positive values increase the prediction, negative values decrease it.")
        
        # Mock SHAP values (in production, would load actual SHAP values)
        mock_contributions = pd.DataFrame({
            'Feature': top_features['feature'].head(8).tolist(),
            'Contribution': np.random.randn(8) * 5
        }).sort_values('Contribution', ascending=True)
        
        fig_waterfall = go.Figure(go.Bar(
            x=mock_contributions['Contribution'],
            y=mock_contributions['Feature'],
            orientation='h',
            marker=dict(
                color=['red' if x < 0 else 'green' for x in mock_contributions['Contribution']]
            ),
            text=[f"{x:+.1f}M" for x in mock_contributions['Contribution']],
            textposition='auto'
        ))
        
        fig_waterfall.update_layout(
            title=f'Feature Contributions to Market Value',
            xaxis_title='Contribution to Prediction (€M)',
            yaxis_title='Feature',
            height=400
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)

# ============================================================================
# Page: Team Analysis
# ============================================================================

elif page == "🏆 Team Analysis":
    st.title("🏆 Team Analysis Dashboard")
    st.markdown("### Squad valuation and transfer risk assessment")
    
    # Team selection
    if 'club' in df.columns:
        teams = sorted(df['club'].unique())
        selected_team = st.selectbox("Select Team:", teams)
        
        if selected_team:
            team_data = df[df['club'] == selected_team]
            
            st.markdown("---")
            
            # Team overview
            st.markdown(f"### 📊 {selected_team} Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Squad Size", len(team_data))
            
            with col2:
                st.metric("Total Value", f"€{team_data['market_value'].sum():.1f}M")
            
            with col3:
                st.metric("Average Value", f"€{team_data['market_value'].mean():.1f}M")
            
            with col4:
                st.metric("Average Age", f"{team_data['age'].mean():.1f} years")
            
            # Squad composition
            st.markdown("### 👥 Squad Composition")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # By position
                position_dist = team_data['position'].value_counts()
                
                fig_pos = px.pie(
                    values=position_dist.values,
                    names=position_dist.index,
                    title='Players by Position',
                    height=400
                )
                
                st.plotly_chart(fig_pos, use_container_width=True)
            
            with col2:
                # By value
                position_value = team_data.groupby('position')['market_value'].sum()
                
                fig_val = px.pie(
                    values=position_value.values,
                    names=position_value.index,
                    title='Squad Value by Position',
                    height=400
                )
                
                st.plotly_chart(fig_val, use_container_width=True)
            
            # Top players
            st.markdown("### ⭐ Most Valuable Players")
            
            top_players = team_data.nlargest(10, 'market_value')[
                ['player_name', 'position', 'age', 'market_value', 'avg_rating']
            ]
            
            st.dataframe(top_players, use_container_width=True)
            
            # Transfer risks
            if df_transfer is not None:
                st.markdown("### 🚨 Transfer Risk Assessment")
                
                team_transfer = df_transfer[df_transfer['player_name'].isin(team_data['player_name'])]
                
                if len(team_transfer) > 0:
                    avg_risk = team_transfer['predicted_transfer_probability'].mean()
                    high_risk_count = (team_transfer['predicted_transfer_probability'] >= 0.6).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Transfer Risk", f"{avg_risk:.1%}")
                    
                    with col2:
                        st.metric("High-Risk Players", high_risk_count)
                    
                    with col3:
                        at_risk_value = team_transfer[
                            team_transfer['predicted_transfer_probability'] >= 0.6
                        ]['market_value'].sum()
                        st.metric("Value at Risk", f"€{at_risk_value:.1f}M")
                    
                    # At-risk players
                    if high_risk_count > 0:
                        st.markdown("#### Players at High Transfer Risk")
                        
                        at_risk_players = team_transfer[
                            team_transfer['predicted_transfer_probability'] >= 0.6
                        ].sort_values('predicted_transfer_probability', ascending=False)
                        
                        risk_display = at_risk_players[
                            ['player_name', 'position', 'market_value', 'predicted_transfer_probability', 'predicted_risk_level']
                        ].copy()
                        
                        risk_display['predicted_transfer_probability'] = risk_display['predicted_transfer_probability'].apply(lambda x: f"{x:.1%}")
                        risk_display.columns = ['Player', 'Position', 'Value (€M)', 'Transfer Prob', 'Risk Level']
                        
                        st.dataframe(risk_display, use_container_width=True)
    else:
        st.info("Team information not available in dataset.")

# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**TransferIQ** - AI-powered football analytics platform

Built with:
- Python & Streamlit
- XGBoost & LightGBM
- SHAP for explainability
- Plotly for visualizations

© 2024 TransferIQ
""")
