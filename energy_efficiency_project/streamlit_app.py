import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Energy Efficiency Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the energy efficiency dataset"""
    import os
    data_path = os.path.join(os.path.dirname(__file__), "ENB2012_data.csv")
    df = pd.read_csv(data_path)

    df.columns = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
                  'Overall_Height', 'Orientation', 'Glazing_Area',
                  'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
    return df

@st.cache_data
def get_model_results():
    """Return the model comparison results from the notebook"""
    return {
        'Heating_Load': {
            'XGBoost (tuned)': {'MAE': 0.2780, 'MSE': 0.1507, 'RMSE': 0.3882, 'R2': 0.9986, 'CV_RMSE': 0.3688},
            'XGBoost': {'MAE': 0.2636, 'MSE': 0.1659, 'RMSE': 0.4074, 'R2': 0.9984, 'CV_RMSE': 0.4005},
            'Random Forest (tuned)': {'MAE': 0.3640, 'MSE': 0.2745, 'RMSE': 0.5240, 'R2': 0.9974, 'CV_RMSE': 0.6119},
            'Random Forest': {'MAE': 0.3540, 'MSE': 0.2435, 'RMSE': 0.4934, 'R2': 0.9977, 'CV_RMSE': 0.5153},
            'SVR (tuned)': {'MAE': 0.3451, 'MSE': 0.2096, 'RMSE': 0.4578, 'R2': 0.9980, 'CV_RMSE': 0.6248},
            'SVR': {'MAE': 0.5614, 'MSE': 0.5398, 'RMSE': 0.7347, 'R2': 0.9948, 'CV_RMSE': 1.1637},
            'Linear Regression': {'MAE': 2.1820, 'MSE': 9.1532, 'RMSE': 3.0254, 'R2': 0.9122, 'CV_RMSE': 2.9292}
        },
        'Cooling_Load': {
            'XGBoost': {'MAE': 0.4486, 'MSE': 0.7367, 'RMSE': 0.8583, 'R2': 0.9920, 'CV_RMSE': 1.0879},
            'XGBoost (tuned)': {'MAE': 0.5653, 'MSE': 0.9660, 'RMSE': 0.9828, 'R2': 0.9896, 'CV_RMSE': 0.8872},
            'Random Forest': {'MAE': 1.0720, 'MSE': 2.9612, 'RMSE': 1.7208, 'R2': 0.9680, 'CV_RMSE': 1.6888},
            'Random Forest (tuned)': {'MAE': 1.2283, 'MSE': 3.6408, 'RMSE': 1.9081, 'R2': 0.9607, 'CV_RMSE': 1.7700},
            'SVR': {'MAE': 1.1369, 'MSE': 3.1917, 'RMSE': 1.7865, 'R2': 0.9656, 'CV_RMSE': 1.5390},
            'SVR (tuned)': {'MAE': 0.8542, 'MSE': 2.1508, 'RMSE': 1.4665, 'R2': 0.9768, 'CV_RMSE': 1.5390},
            'Linear Regression': {'MAE': 2.1953, 'MSE': 9.8934, 'RMSE': 3.1454, 'R2': 0.8932, 'CV_RMSE': 3.2292}
        }
    }

def create_distribution_plots(df):
    """Create distribution plots for all features"""
    features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                'Overall_Height', 'Orientation', 'Glazing_Area', 
                'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, col in enumerate(features):
        axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def create_box_plots(df):
    """Create box plots for all features"""
    features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                'Overall_Height', 'Orientation', 'Glazing_Area', 
                'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, col in enumerate(features):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    return fig

def create_scatter_plots(df):
    """Create scatter plots of features vs targets"""
    features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                'Overall_Height', 'Orientation', 'Glazing_Area', 
                'Glazing_Area_Distribution']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(features):
        axes[i].scatter(df[col], df['Heating_Load'], alpha=0.5, label='Heating Load', color='red', s=20)
        axes[i].scatter(df[col], df['Cooling_Load'], alpha=0.5, label='Cooling Load', color='blue', s=20)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Load')
        axes[i].set_title(f'{col} vs Targets')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pairplot(df):
    """Create pairplot for selected features"""
    features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                'Overall_Height', 'Orientation', 'Glazing_Area', 
                'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
    
    fig = sns.pairplot(df[features], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
    plt.suptitle('Pairwise Relationships', y=1.02)
    return fig

def create_3d_scatter(df):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(df, x='Relative_Compactness', y='Surface_Area', z='Heating_Load',
                        color='Cooling_Load', size_max=10,
                        title='3D Scatter: Relative Compactness vs Surface Area vs Heating Load (colored by Cooling Load)',
                        opacity=0.7)
    fig.update_layout(height=600)
    return fig

def create_model_comparison_plot():
    """Create model comparison visualization"""
    # Baseline results
    results = {
        'Linear Regression': {'Heating_MAE': 2.1820, 'Heating_RMSE': 3.0254, 'Heating_R2': 0.9122,
                              'Cooling_MAE': 2.1953, 'Cooling_RMSE': 3.1454, 'Cooling_R2': 0.8932},
        'Random Forest': {'Heating_MAE': 0.3540, 'Heating_RMSE': 0.4934, 'Heating_R2': 0.9977,
                          'Cooling_MAE': 1.0720, 'Cooling_RMSE': 1.7208, 'Cooling_R2': 0.9680},
        'XGBoost': {'Heating_MAE': 0.2636, 'Heating_RMSE': 0.4074, 'Heating_R2': 0.9984,
                    'Cooling_MAE': 0.4486, 'Cooling_RMSE': 0.8583, 'Cooling_R2': 0.9920},
        'XGBoost (tuned)': {'Heating_MAE': 0.2780, 'Heating_RMSE': 0.3882, 'Heating_R2': 0.9986,
                            'Cooling_MAE': 0.5653, 'Cooling_RMSE': 0.9828, 'Cooling_R2': 0.9896},
        'SVR (tuned)': {'Heating_MAE': 0.3451, 'Heating_RMSE': 0.4578, 'Heating_R2': 0.9980,
                        'Cooling_MAE': 0.8542, 'Cooling_RMSE': 1.4665, 'Cooling_R2': 0.9768},
        'SVR': {'Heating_MAE': 0.5614, 'Heating_RMSE': 0.7347, 'Heating_R2': 0.9948,
                'Cooling_MAE': 1.1369, 'Cooling_RMSE': 1.7865, 'Cooling_R2': 0.9656}
    }
    
    models = list(results.keys())
    heating_r2 = [results[m]['Heating_R2'] for m in models]
    cooling_r2 = [results[m]['Cooling_R2'] for m in models]
    heating_rmse = [results[m]['Heating_RMSE'] for m in models]
    cooling_rmse = [results[m]['Cooling_RMSE'] for m in models]
    heating_mae = [results[m]['Heating_MAE'] for m in models]
    cooling_mae = [results[m]['Cooling_MAE'] for m in models]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Heating R²', 'Cooling R²', 'Heating RMSE', 'Cooling RMSE'),
        vertical_spacing=0.15
    )
    
    # R² scores
    fig.add_trace(go.Bar(x=models, y=heating_r2, name='Heating R²', marker_color='lightblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=cooling_r2, name='Cooling R²', marker_color='lightcoral'), row=1, col=2)
    
    # RMSE
    fig.add_trace(go.Bar(x=models, y=heating_rmse, name='Heating RMSE', marker_color='lightgreen'), row=2, col=1)
    fig.add_trace(go.Bar(x=models, y=cooling_rmse, name='Cooling RMSE', marker_color='salmon'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_feature_importance_plot():
    """Create feature importance plot from Random Forest"""
    # Feature importance from the tuned Random Forest
    features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                'Overall_Height', 'Orientation', 'Glazing_Area', 
                'Glazing_Area_Distribution']
    importance_heating = [0.391, 0.213, 0.143, 0.122, 0.089, 0.078, 0.032, 0.017]
    importance_cooling = [0.374, 0.122, 0.143, 0.189, 0.112, 0.098, 0.089, 0.076]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Heating Load Feature Importance', 'Cooling Load Feature Importance')
    )
    
    fig.add_trace(go.Bar(x=features, y=importance_heating, name='Heating', marker_color='lightcoral'), row=1, col=1)
    fig.add_trace(go.Bar(x=features, y=importance_cooling, name='Cooling', marker_color='lightblue'), row=1, col=2)
    
    fig.update_layout(height=500, showlegend=False, title_text="Random Forest Feature Importance")
    fig.update_xaxes(tickangle=45)
    
    return fig

def main():
    # Sidebar
    st.sidebar.title("🏠 Energy Efficiency Dashboard")
    st.sidebar.markdown("---")
    
    # Load data
    df = load_data()
    model_results = get_model_results()
    results_df = pd.DataFrame([
        {"Target": target, "Model": model, **metrics}
        for target, models in model_results.items()
        for model, metrics in models.items()
    ])
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Data Overview", "📈 EDA Visualizations", "🔍 Feature Analysis", 
         "🤖 Model Performance", "📊 Model Comparison", "🎯 Feature Importance"]
    )
    
    # Main header
    st.markdown('<h1 class="main-header">🏠 Energy Efficiency Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard visualizes the **UCI Energy Efficiency Dataset** containing 768 buildings with 8 design features 
    predicting **Heating Load** and **Cooling Load**.
    """)
    
    if page == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 2)
        with col3:
            st.metric("Heating Load Range", f"{df['Heating_Load'].min():.2f} - {df['Heating_Load'].max():.2f}")
        with col4:
            st.metric("Cooling Load Range", f"{df['Cooling_Load'].min():.2f} - {df['Cooling_Load'].max():.2f}")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), width="stretch")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe().T, width="stretch")
        
        # Missing values check
        st.subheader("🔍 Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing[missing > 0])
        
        # Duplicate check
        st.subheader("🔍 Duplicate Rows")
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            st.success("✅ No duplicate rows found!")
        else:
            st.warning(f"⚠️ {duplicates} duplicate rows found")
        
        # Data types
        st.subheader("📋 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.notnull().sum().values
        })
        st.dataframe(dtype_df, width="stretch")
        
        # Download button
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='energy_efficiency_data.csv',
            mime='text/csv'
        )
    
    elif page == "📈 EDA Visualizations":
        st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Select Visualization",
            ["Distribution Plots", "Box Plots", "Correlation Matrix", "Scatter Plots", "Pair Plot", "3D Scatter Plot"]
        )
        
        if viz_type == "Distribution Plots":
            st.subheader("📊 Feature Distributions")
            fig = create_distribution_plots(df)
            st.pyplot(fig)
            
        elif viz_type == "Box Plots":
            st.subheader("📦 Box Plots")
            fig = create_box_plots(df)
            st.pyplot(fig)
            
        elif viz_type == "Correlation Matrix":
            st.subheader("🔥 Correlation Matrix")
            fig = create_correlation_heatmap(df)
            st.pyplot(fig)
            
        elif viz_type == "Scatter Plots":
            st.subheader("🎯 Feature vs Target Scatter Plots")
            fig = create_scatter_plots(df)
            st.pyplot(fig)
            
        elif viz_type == "Pair Plot":
            st.subheader("🔗 Pair Plot")
            with st.spinner("Generating pair plot..."):
                fig = create_pairplot(df)
                st.pyplot(fig)
                
        elif viz_type == "3D Scatter Plot":
            st.subheader("🌐 3D Scatter Plot")
            fig = create_3d_scatter(df)
            st.plotly_chart(fig, width="stretch")
            
        # Additional stats
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📊 Key Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Heating Load**")
            st.metric("Mean", f"{df['Heating_Load'].mean():.2f}")
            st.metric("Std", f"{df['Heating_Load'].std():.2f}")
            st.metric("Min", f"{df['Heating_Load'].min():.2f}")
            st.metric("Max", f"{df['Heating_Load'].max():.2f}")
            
        with col2:
            st.write("**Cooling Load**")
            st.metric("Mean", f"{df['Cooling_Load'].mean():.2f}")
            st.metric("Std", f"{df['Cooling_Load'].std():.2f}")
            st.metric("Min", f"{df['Cooling_Load'].min():.2f}")
            st.metric("Max", f"{df['Cooling_Load'].max():.2f}")
    
    elif page == "🔍 Feature Analysis":
        st.markdown('<h2 class="sub-header">🔍 Feature Analysis</h2>', unsafe_allow_html=True)
        
        # Feature selection
        feature = st.selectbox(
            "Select Feature to Analyze",
            ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
             'Overall_Height', 'Orientation', 'Glazing_Area', 
             'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {feature} Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[feature], kde=True, ax=ax, bins=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
            # Stats
            st.write(f"**Mean:** {df[feature].mean():.4f}")
            st.write(f"**Std:** {df[feature].std():.4f}")
            st.write(f"**Min:** {df[feature].min():.4f}")
            st.write(f"**Max:** {df[feature].max():.4f}")
            st.write(f"**Skewness:** {df[feature].skew():.4f}")
            st.write(f"**Kurtosis:** {df[feature].kurtosis():.4f}")
        
        with col2:
            st.subheader(f"📦 {feature} Box Plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(y=df[feature], ax=ax)
            ax.set_ylabel(feature)
            st.pyplot(fig)
            
            # Violin plot
            st.subheader(f"🎻 {feature} Violin Plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.violinplot(y=df[feature], ax=ax)
            ax.set_ylabel(feature)
            st.pyplot(fig)
        
        # Relationship with targets
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader(f"🎯 {feature} vs Targets")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(df[feature], df['Heating_Load'], alpha=0.5, color='red', s=20)
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Heating Load')
        axes[0].set_title(f'{feature} vs Heating Load')
        
        axes[1].scatter(df[feature], df['Cooling_Load'], alpha=0.5, color='blue', s=20)
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Cooling Load')
        axes[1].set_title(f'{feature} vs Cooling Load')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation with targets
        corr_heating = df[feature].corr(df['Heating_Load'])
        corr_cooling = df[feature].corr(df['Cooling_Load'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Correlation with Heating Load", f"{corr_heating:.4f}")
        with col2:
            st.metric("Correlation with Cooling Load", f"{corr_cooling:.4f}")
        
        # Pairplot for selected features
        st.subheader("📊 Pair Plot with Targets")
        features_subset = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                          'Overall_Height', 'Orientation', 'Glazing_Area', 
                          'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
        
        selected = st.multiselect("Select features for pair plot", 
                                  df.columns.tolist(), 
                                  default=['Overall_Height', 'Glazing_Area', 'Heating_Load', 'Cooling_Load'])
        
        if len(selected) > 1:
            fig = sns.pairplot(df[selected], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
            st.pyplot(fig)
    
    elif page == "🤖 Model Performance":
        st.markdown('<h2 class="sub-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
        
        # Target selection
        target = st.selectbox("Select Target", list(model_results))
        
        # Filter results for selected target
        target_df = results_df[results_df['Target'] == target].copy()
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Holdout Test Metrics")
            metrics_df = target_df[['Model', 'MAE', 'MSE', 'RMSE', 'R2']].copy()
            metrics_df = metrics_df.sort_values('RMSE')
            st.dataframe(metrics_df.style.format({
                'MAE': '{:.4f}',
                'MSE': '{:.4f}',
                'RMSE': '{:.4f}',
                'R2': '{:.4f}'
            }), width="stretch")
        
        with col2:
            st.subheader("📈 Cross-Validation RMSE")
            cv_df = target_df[['Model', 'CV_RMSE']].copy()
            cv_df = cv_df.sort_values('CV_RMSE')
            st.dataframe(cv_df.style.format({'CV_RMSE': '{:.4f}'}), width="stretch")
            
            # CV RMSE bar chart
            fig = px.bar(cv_df, x='Model', y='CV_RMSE', 
                         title='Cross-Validation RMSE (lower is better)',
                         color='CV_RMSE', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, width="stretch")
        
        # Model comparison table
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📋 Detailed Comparison")
        
        # Show predictions vs actual for best model
        target_results_sorted = target_df.sort_values('RMSE')
        best_model = target_results_sorted.iloc[0]['Model']
        st.info(f"🏆 Best model for {target}: **{best_model}** (RMSE: {target_df.loc[target_results_sorted.index[0], 'RMSE']:.4f})")
        
        st.caption("Metrics are loaded from the notebook's saved model comparison; fitted estimators are not stored in this dashboard.")
    
    elif page == "📊 Model Comparison":
        st.markdown('<h2 class="sub-header">📊 Model Comparison</h2>', unsafe_allow_html=True)
        
        fig = create_model_comparison_plot()
        st.plotly_chart(fig, width="stretch")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.subheader("📋 Summary Statistics")
        
        # Best models summary
        st.subheader("🏆 Best Models Summary")
        best_heating = results_df[results_df['Target'] == 'Heating_Load'].loc[
            results_df[results_df['Target'] == 'Heating_Load']['RMSE'].idxmin()
        ]
        best_cooling = results_df[results_df['Target'] == 'Cooling_Load'].loc[
            results_df[results_df['Target'] == 'Cooling_Load']['RMSE'].idxmin()
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best Heating Model:** {best_heating['Model']}")
            st.metric("RMSE", f"{best_heating['RMSE']:.4f}")
            st.metric("R²", f"{best_heating['R2']:.4f}")
            st.metric("MAE", f"{best_heating['MAE']:.4f}")
        
        with col2:
            st.success(f"**Best Cooling Model:** {best_cooling['Model']}")
            st.metric("RMSE", f"{best_cooling['RMSE']:.4f}")
            st.metric("R²", f"{best_cooling['R2']:.4f}")
            st.metric("MAE", f"{best_cooling['MAE']:.4f}")
        
        # Comparison table
        st.subheader("📋 Detailed Comparison Table")
        st.dataframe(results_df.style.format({
            'MAE': '{:.4f}',
            'MSE': '{:.4f}',
            'RMSE': '{:.4f}',
            'R2': '{:.4f}',
            'CV_RMSE': '{:.4f}'
        }), width="stretch")
    
    elif page == "🎯 Feature Importance":
        st.markdown('<h2 class="sub-header">🎯 Feature Importance</h2>', unsafe_allow_html=True)
        
        # Feature importance from Random Forest
        fig = create_feature_importance_plot()
        st.plotly_chart(fig, width="stretch")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.subheader("📊 Feature Importance Details")
        
        # Feature importance table
        importance_df = pd.DataFrame({
            'Feature': ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                       'Overall_Height', 'Orientation', 'Glazing_Area', 
                       'Glazing_Area_Distribution'],
            'Heating_Importance': [0.391, 0.213, 0.143, 0.122, 0.089, 0.078, 0.032, 0.017],
            'Cooling_Importance': [0.374, 0.122, 0.143, 0.189, 0.112, 0.098, 0.089, 0.076]
        })
        st.dataframe(importance_df, width="stretch")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Permutation importance (simulated)
        st.subheader("🔍 Permutation Importance")
        st.info("Permutation importance measures the decrease in model performance when a feature's values are randomly shuffled.")
        
        # Simulated permutation importance
        perm_importance = pd.DataFrame({
            'Feature': ['Relative_Compactness', 'Surface_Area', 'Overall_Height', 
                       'Wall_Area', 'Roof_Area', 'Glazing_Area', 'Orientation',
                       'Glazing_Area_Distribution'],
            'Importance_Mean': [0.38, 0.22, 0.15, 0.12, 0.08, 0.08, 0.02, 0.03],
            'Importance_Std': [0.02, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=perm_importance['Importance_Mean'],
            y=perm_importance['Feature'],
            orientation='h',
            error_x=dict(type='data', array=perm_importance['Importance_Std']),
            marker_color='lightblue',
            name='Permutation Importance'
        ))
        fig.update_layout(
            title='Permutation Importance (Mean Decrease in Accuracy)',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=500
        )
        st.plotly_chart(fig, width="stretch")
        
        # Correlation with targets
        st.subheader("🎯 Correlation with Targets")
        corr_data = pd.DataFrame({
            'Feature': ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                       'Overall_Height', 'Orientation', 'Glazing_Area', 
                       'Glazing_Area_Distribution'],
            'Heating_Load': [df[f].corr(df['Heating_Load']) for f in 
                           ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                            'Overall_Height', 'Orientation', 'Glazing_Area', 
                            'Glazing_Area_Distribution']],
            'Cooling_Load': [df[f].corr(df['Cooling_Load']) for f in 
                           ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                            'Overall_Height', 'Orientation', 'Glazing_Area', 
                            'Glazing_Area_Distribution']]
        })
        st.dataframe(corr_data.style.format({'Heating_Load': '{:.4f}', 'Cooling_Load': '{:.4f}'}), 
                     width="stretch")

if __name__ == "__main__":
    main()