"""
Visualization components for dashboard.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from src.dashboard.config import SENTIMENT_COLORS


def create_confidence_gauge(confidence: float, sentiment: str) -> go.Figure:
    """
    Create confidence gauge chart.

    Args:
        confidence: Confidence score (0-1)
        sentiment: Sentiment label

    Returns:
        Plotly figure
    """
    color = SENTIMENT_COLORS.get(sentiment.lower(), "#808080")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence", "font": {"size": 20}},
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "rgba(255, 255, 255, 0.1)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 75, 75, 0.3)"},
                    {"range": [50, 75], "color": "rgba(255, 165, 0, 0.3)"},
                    {"range": [75, 100], "color": "rgba(0, 210, 106, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": confidence * 100,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_probability_bars(probabilities: Dict[str, float]) -> go.Figure:
    """
    Create probability distribution bar chart.

    Args:
        probabilities: Dictionary of sentiment probabilities

    Returns:
        Plotly figure
    """
    sentiments = list(probabilities.keys())
    values = [probabilities[s] * 100 for s in sentiments]
    colors = [SENTIMENT_COLORS.get(s.lower(), "#808080") for s in sentiments]

    fig = go.Figure(
        data=[
            go.Bar(
                x=sentiments,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                textfont=dict(size=16, color="white"),
            )
        ]
    )

    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    return fig


def create_sentiment_pie(sentiment_counts: Dict[str, int]) -> go.Figure:
    """
    Create sentiment distribution pie chart.

    Args:
        sentiment_counts: Dictionary of sentiment counts

    Returns:
        Plotly figure
    """
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    colors = [SENTIMENT_COLORS.get(label.lower(), "#808080") for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo="label+percent",
                textfont=dict(size=14, color="white"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Sentiment Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


def create_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Create confidence score histogram.

    Args:
        df: DataFrame with predictions

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for sentiment in df["sentiment"].unique():
        sentiment_data = df[df["sentiment"] == sentiment]["confidence"]
        color = SENTIMENT_COLORS.get(sentiment.lower(), "#808080")

        fig.add_trace(
            go.Histogram(
                x=sentiment_data,
                name=sentiment.capitalize(),
                marker_color=color,
                opacity=0.7,
                nbinsx=20,
            )
        )

    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    return fig


def create_predictions_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create predictions over time line chart.

    Args:
        df: DataFrame with predictions

    Returns:
        Plotly figure
    """
    # Group by date and sentiment
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    timeline = df.groupby(["date", "sentiment"]).size().reset_index(name="count")

    fig = go.Figure()

    for sentiment in timeline["sentiment"].unique():
        sentiment_data = timeline[timeline["sentiment"] == sentiment]
        color = SENTIMENT_COLORS.get(sentiment.lower(), "#808080")

        fig.add_trace(
            go.Scatter(
                x=sentiment_data["date"],
                y=sentiment_data["count"],
                name=sentiment.capitalize(),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title="Predictions Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Predictions",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    return fig


def create_model_comparison_bar(
    model_metrics: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Create model comparison bar chart.

    Args:
        model_metrics: Dictionary of model metrics

    Returns:
        Plotly figure
    """
    models = list(model_metrics.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig = go.Figure()

    colors = ["#FF4B4B", "#00D26A", "#FFA500", "#1E90FF", "#9370DB"]

    for i, metric in enumerate(metrics):
        values = [model_metrics[model].get(metric, 0) * 100 for model in models]

        fig.add_trace(
            go.Bar(
                name=metric.upper().replace("_", " "),
                x=models,
                y=values,
                marker_color=colors[i],
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    return fig


def create_text_length_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create text length distribution histogram.

    Args:
        df: DataFrame with text data

    Returns:
        Plotly figure
    """
    df["text_length"] = df["text"].str.len()

    fig = go.Figure(
        data=[
            go.Histogram(
                x=df["text_length"], nbinsx=50, marker_color="#667eea", opacity=0.8
            )
        ]
    )

    fig.update_layout(
        title="Text Length Distribution",
        xaxis_title="Text Length (characters)",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    return fig
