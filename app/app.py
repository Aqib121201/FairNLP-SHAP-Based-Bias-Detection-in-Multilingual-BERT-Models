"""
FairNLP Dashboard

A Streamlit-based dashboard for exploring bias detection results
in multilingual BERT models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config

# Page configuration
st.set_page_config(
    page_title="FairNLP Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .bias-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .fairness-success {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration."""
    return Config()

@st.cache_data
def load_reports():
    """Load all bias reports."""
    config = load_config()
    reports_dir = Path(config.reports_dir)
    
    reports = {}
    if reports_dir.exists():
        for report_file in reports_dir.glob("*_bias_report.json"):
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                # Extract task and attribute from filename
                filename = report_file.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    task = parts[0]
                    attr = parts[1]
                    
                    if task not in reports:
                        reports[task] = {}
                    reports[task][attr] = report_data
            except Exception as e:
                st.error(f"Error loading report {report_file}: {e}")
    
    return reports

@st.cache_data
def load_final_report():
    """Load final comprehensive report."""
    config = load_config()
    final_report_path = Path(config.reports_dir) / "final_report.json"
    
    if final_report_path.exists():
        with open(final_report_path, 'r') as f:
            return json.load(f)
    return None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 FairNLP Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### SHAP-Based Bias Detection in Multilingual BERT Models")
    
    # Load data
    config = load_config()
    reports = load_reports()
    final_report = load_final_report()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Bias Analysis", "Model Performance", "SHAP Visualizations", "Reports"]
    )
    
    if page == "Overview":
        show_overview(config, final_report)
    elif page == "Bias Analysis":
        show_bias_analysis(reports)
    elif page == "Model Performance":
        show_model_performance(final_report)
    elif page == "SHAP Visualizations":
        show_shap_visualizations(config)
    elif page == "Reports":
        show_reports(config)

def show_overview(config, final_report):
    """Show overview page."""
    st.header("📊 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Information")
        st.info("""
        **FairNLP** is a comprehensive framework for detecting and analyzing bias 
        in multilingual language models using SHAP values and fairness metrics.
        
        - **Languages**: English, German, Hindi
        - **Tasks**: Sentiment Analysis, Translation
        - **Models**: mBERT, English BERT, German BERT, Hindi BERT
        - **Fairness Metrics**: Demographic Parity, Equalized Odds, KL Divergence
        """)
    
    with col2:
        st.subheader("Key Metrics")
        if final_report and 'summary' in final_report:
            summary = final_report['summary']
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Models Trained", summary.get('total_models_trained', 0))
                st.metric("Bias Analyses", summary.get('total_bias_analyses', 0))
            
            with col2b:
                st.metric("High Bias Models", summary.get('high_bias_models', 0))
                st.metric("Recommendations", len(summary.get('recommendations', [])))
        else:
            st.warning("No final report available")
    
    # Configuration summary
    st.subheader("Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Model Configuration**")
        st.json(config.model.__dict__)
    
    with col4:
        st.write("**Fairness Configuration**")
        st.json(config.fairness.__dict__)
    
    # Recommendations
    if final_report and 'summary' in final_report:
        recommendations = final_report['summary'].get('recommendations', [])
        if recommendations:
            st.subheader("🚨 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

def show_bias_analysis(reports):
    """Show bias analysis page."""
    st.header("🔍 Bias Analysis")
    
    if not reports:
        st.warning("No bias reports available. Run the analysis pipeline first.")
        return
    
    # Task selection
    task = st.selectbox("Select Task", list(reports.keys()))
    
    if task in reports:
        task_reports = reports[task]
        
        # Attribute selection
        attr = st.selectbox("Select Demographic Attribute", list(task_reports.keys()))
        
        if attr in task_reports:
            report = task_reports[attr]
            
            # Summary metrics
            st.subheader(f"Bias Summary - {task.title()} ({attr})")
            
            if 'summary' in report:
                summary = report['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Groups", summary.get('total_groups', 0))
                
                with col2:
                    bias_score = summary.get('mean_bias_score', 0)
                    st.metric("Mean Bias Score", f"{bias_score:.4f}")
                
                with col3:
                    threshold_exceeded = summary.get('bias_threshold_exceeded', 0)
                    st.metric("Threshold Exceeded", threshold_exceeded)
                
                with col4:
                    std_bias = summary.get('std_bias_score', 0)
                    st.metric("Bias Std Dev", f"{std_bias:.4f}")
                
                # Bias score visualization
                st.subheader("Bias Score Distribution")
                
                if 'detailed_analysis' in report:
                    detailed = report['detailed_analysis']
                    
                    groups = list(detailed.keys())
                    bias_scores = [detailed[g]['bias_score'] for g in groups]
                    
                    fig = px.bar(
                        x=groups,
                        y=bias_scores,
                        title=f"Bias Scores by {attr}",
                        labels={'x': attr.title(), 'y': 'Bias Score'},
                        color=bias_scores,
                        color_continuous_scale='RdYlBu_r'
                    )
                    
                    # Add threshold line
                    fig.add_hline(
                        y=0.1,  # Default threshold
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Bias Threshold"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed analysis
                st.subheader("Detailed Analysis")
                
                if 'detailed_analysis' in report:
                    detailed = report['detailed_analysis']
                    
                    for group, analysis in detailed.items():
                        with st.expander(f"Group: {group}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Bias Score:**", f"{analysis['bias_score']:.4f}")
                                st.write("**Bias Indicators:**", analysis['bias_indicators_count'])
                                st.write("**Bias Types:**", ", ".join(analysis['bias_types']))
                            
                            with col2:
                                if 'feature_importance' in analysis:
                                    fi = analysis['feature_importance']
                                    st.write("**Feature Importance:**")
                                    st.write(f"- Mean: {fi.get('mean_importance', 0):.4f}")
                                    st.write(f"- Std: {fi.get('std_importance', 0):.4f}")
                                    st.write(f"- Max: {fi.get('max_importance', 0):.4f}")
                
                # Recommendations
                if 'recommendations' in report:
                    st.subheader("Recommendations")
                    for i, rec in enumerate(report['recommendations'], 1):
                        st.info(f"**{i}.** {rec}")

def show_model_performance(final_report):
    """Show model performance page."""
    st.header("📈 Model Performance")
    
    if not final_report:
        st.warning("No final report available. Run the training pipeline first.")
        return
    
    if 'training_results' not in final_report:
        st.warning("No training results available.")
        return
    
    training_results = final_report['training_results']
    
    # Task selection
    task = st.selectbox("Select Task", list(training_results.keys()))
    
    if task in training_results:
        task_results = training_results[task]
        
        st.subheader(f"Performance Metrics - {task.title()}")
        
        # Create performance comparison table
        performance_data = []
        
        for model_name, results in task_results.items():
            if 'test_results' in results and 'metrics' in results['test_results']:
                metrics = results['test_results']['metrics']
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'AUC': metrics.get('auc', 0)
                })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            st.dataframe(df, use_container_width=True)
            
            # Performance visualization
            st.subheader("Performance Comparison")
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
            
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df['Model'],
                    y=df[metric],
                    text=[f"{val:.3f}" for val in df[metric]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title=f"{task.title()} Model Performance Comparison",
                barmode='group',
                xaxis_title="Model",
                yaxis_title="Score",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No performance metrics available")

def show_shap_visualizations(config):
    """Show SHAP visualizations page."""
    st.header("📊 SHAP Visualizations")
    
    # Load visualization files
    viz_dir = Path(config.visualizations_dir)
    
    if not viz_dir.exists():
        st.warning("No visualizations available. Run the analysis pipeline first.")
        return
    
    # Get available visualizations
    viz_files = list(viz_dir.glob("*.html"))
    
    if not viz_files:
        st.warning("No visualization files found.")
        return
    
    # Group visualizations by task and attribute
    viz_groups = {}
    for viz_file in viz_files:
        filename = viz_file.stem
        parts = filename.split('_')
        if len(parts) >= 3:
            task = parts[0]
            attr = parts[1]
            viz_type = parts[2]
            
            if task not in viz_groups:
                viz_groups[task] = {}
            if attr not in viz_groups[task]:
                viz_groups[task][attr] = {}
            
            viz_groups[task][attr][viz_type] = viz_file
    
    # Task selection
    task = st.selectbox("Select Task", list(viz_groups.keys()))
    
    if task in viz_groups:
        task_viz = viz_groups[task]
        
        # Attribute selection
        attr = st.selectbox("Select Demographic Attribute", list(task_viz.keys()))
        
        if attr in task_viz:
            attr_viz = task_viz[attr]
            
            # Visualization type selection
            viz_type = st.selectbox("Select Visualization Type", list(attr_viz.keys()))
            
            if viz_type in attr_viz:
                viz_file = attr_viz[viz_type]
                
                st.subheader(f"{viz_type.replace('_', ' ').title()} - {task.title()} ({attr})")
                
                # Display HTML visualization
                with open(viz_file, 'r') as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=600)

def show_reports(config):
    """Show reports page."""
    st.header("📋 Reports")
    
    reports_dir = Path(config.reports_dir)
    
    if not reports_dir.exists():
        st.warning("No reports directory found.")
        return
    
    # List available reports
    report_files = list(reports_dir.glob("*.json"))
    
    if not report_files:
        st.warning("No report files found.")
        return
    
    st.subheader("Available Reports")
    
    for report_file in report_files:
        with st.expander(f"📄 {report_file.name}"):
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                # Show report summary
                if 'summary' in report_data:
                    summary = report_data['summary']
                    st.write("**Summary:**")
                    st.json(summary)
                
                # Show full report
                st.write("**Full Report:**")
                st.json(report_data)
                
                # Download button
                st.download_button(
                    label=f"Download {report_file.name}",
                    data=json.dumps(report_data, indent=2),
                    file_name=report_file.name,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error loading report: {e}")

if __name__ == "__main__":
    main() 