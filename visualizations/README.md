# Visualizations Directory

This directory contains all generated visualizations for the FairNLP project, including SHAP plots, bias analysis charts, and performance metrics.

## File Naming Convention

Visualization files follow this naming pattern:
```
{task}_{demographic_attribute}_{visualization_type}.{format}
```

### Examples:
- `sentiment_gender_summary_plot.png` - SHAP summary plot for sentiment analysis by gender
- `translation_age_group_comparison_plot.html` - Bias comparison plot for translation by age group
- `sentiment_region_token_plot.png` - Token importance plot for sentiment analysis by region

## Visualization Types

### 1. Summary Plot (`summary_plot`)
- **Purpose**: Shows overall SHAP feature importance across all samples
- **Format**: Heatmap showing feature contributions
- **Use Case**: Identify most important features for model predictions

### 2. Comparison Plot (`comparison_plot`)
- **Purpose**: Compare bias scores across demographic groups
- **Format**: Bar chart with bias scores for each group
- **Use Case**: Identify which groups have highest bias

### 3. Token Plot (`token_plot`)
- **Purpose**: Show importance of specific tokens in bias detection
- **Format**: Bar chart of token importance scores
- **Use Case**: Identify bias-inducing words or phrases

### 4. Parity Plot (`parity_plot`)
- **Purpose**: Show demographic parity violations
- **Format**: Bar chart of parity violation scores
- **Use Case**: Quantify fairness violations across groups

## File Formats

### PNG Files
- Static images suitable for reports and presentations
- High resolution (300 DPI)
- Good for printing and embedding in documents

### HTML Files
- Interactive plots using Plotly
- Can be opened in web browsers
- Allow zooming, hovering, and interaction

## How to Use

### Viewing PNG Files
```bash
# Open with default image viewer
open visualizations/sentiment_gender_summary_plot.png

# Or use command line tools
display visualizations/sentiment_gender_summary_plot.png
```

### Viewing HTML Files
```bash
# Open in web browser
open visualizations/sentiment_gender_summary_plot.html

# Or serve locally
python -m http.server 8000
# Then visit http://localhost:8000/visualizations/
```

### Programmatic Access
```python
import plotly.io as pio
from pathlib import Path

# Load HTML plot
html_file = Path("visualizations/sentiment_gender_summary_plot.html")
with open(html_file, 'r') as f:
    html_content = f.read()

# Display in Jupyter notebook
from IPython.display import HTML
HTML(html_content)
```

## Interpreting Results

### Bias Score Interpretation
- **Low bias (green)**: Score < 0.1 - Model shows minimal bias
- **Medium bias (yellow)**: Score 0.1-0.2 - Some bias detected
- **High bias (red)**: Score > 0.2 - Significant bias requiring attention

### SHAP Value Interpretation
- **Positive values (red)**: Feature contributes to positive prediction
- **Negative values (blue)**: Feature contributes to negative prediction
- **Magnitude**: Higher absolute values indicate stronger influence

### Token Importance
- **High importance**: Tokens that strongly influence model decisions
- **Bias indicators**: Words that may indicate demographic bias
- **Cultural markers**: Terms that reflect cultural or regional bias

## Best Practices

1. **Compare across groups**: Always compare bias scores between demographic groups
2. **Consider context**: Interpret results in the context of your specific use case
3. **Look for patterns**: Identify recurring bias patterns across different attributes
4. **Validate findings**: Cross-reference with domain expertise and additional analysis

## Troubleshooting

### Missing Visualizations
If visualizations are missing:
1. Ensure the analysis pipeline has been run
2. Check that data files exist in the correct format
3. Verify that demographic attributes are present in the data

### Poor Quality Images
For better quality PNG files:
1. Increase DPI setting in configuration
2. Use larger plot dimensions
3. Ensure sufficient data for meaningful visualization

### Interactive Issues
For HTML visualization problems:
1. Check browser compatibility
2. Ensure JavaScript is enabled
3. Try different browsers if issues persist

## Integration with Reports

These visualizations are automatically integrated into:
- Bias analysis reports (`reports/` directory)
- Streamlit dashboard (`app/app.py`)
- Final project report

The dashboard provides an interactive interface to explore all visualizations with filtering and comparison capabilities. 