import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import pandas_ta as pta
import utils.functions as fn
import io
import numpy as np
from typing import Optional, Dict, Any

# =============================================================================
# CENTRALIZED THEME CONFIGURATION
# =============================================================================
# Change all colors and styling from this single location!

class ThemeConfig:
    """Centralized theme configuration for all visualizations"""
    
    # ===== DARK THEME (Default) =====
    DARK = {
        # Background colors
        'bg_color': '#0e1117',
        'paper_bgcolor': '#0e1117',
        'plot_bgcolor': '#1a1d29',
        
        # Text colors
        'text_color': '#ffffff',
        'text_color_secondary': '#b0b0b0',
        
        # Grid and border colors
        'grid_color': '#2a2d3a',
        'grid_color_secondary': '#1e2128',
        'border_color': '#3a3d4a',
        
        # Chart colors - Primary palette
        'primary': '#00d4aa',      # Teal/cyan
        'secondary': '#667eea',    # Blue
        'accent': '#764ba2',      # Purple
        'success': '#00ff88',     # Green
        'warning': '#ffa726',     # Orange
        'danger': '#ff4757',      # Red
        'info': '#00d4ff',        # Light blue
        
        # Chart colors - Extended palette
        'chart_colors': [
            '#00d4aa', '#667eea', '#764ba2', '#00ff88', 
            '#ffa726', '#ff4757', '#00d4ff', '#9b59b6',
            '#e74c3c', '#3498db', '#2ecc71', '#f1c40f'
        ],
        
        # Table colors
        'table_header': '#0078ff',
        'table_row_even': '#1a1e23',
        'table_row_odd': '#262b33',
        
        # Hover colors
        'hover_bg': '#2a2d3a',
        'hover_border': '#00d4aa',
        
        # Shadow and effects
        'shadow_color': 'rgba(0,0,0,0.3)',
        'glow_color': 'rgba(0, 212, 170, 0.2)'
    }
    
    # ===== LIGHT THEME =====
    LIGHT = {
        # Background colors
        'bg_color': '#ffffff',
        'paper_bgcolor': '#ffffff',
        'plot_bgcolor': '#f8f9fa',
        
        # Text colors
        'text_color': '#2c3e50',
        'text_color_secondary': '#7f8c8d',
        
        # Grid and border colors
        'grid_color': '#ecf0f1',
        'grid_color_secondary': '#e9ecef',
        'border_color': '#dee2e6',
        
        # Chart colors - Primary palette
        'primary': '#1f77b4',     # Blue
        'secondary': '#ff7f0e',   # Orange
        'accent': '#2ca02c',      # Green
        'success': '#28a745',     # Green
        'warning': '#ffc107',     # Yellow
        'danger': '#dc3545',      # Red
        'info': '#17a2b8',        # Info blue
        
        # Chart colors - Extended palette
        'chart_colors': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
        ],
        
        # Table colors
        'table_header': '#206fb2',
        'table_row_even': '#f8f9fa',
        'table_row_odd': '#ffffff',
        
        # Hover colors
        'hover_bg': '#e9ecef',
        'hover_border': '#1f77b4',
        
        # Shadow and effects
        'shadow_color': 'rgba(0,0,0,0.1)',
        'glow_color': 'rgba(31, 119, 180, 0.2)'
    }

# =============================================================================
# THEME MANAGER
# =============================================================================

class ThemeManager:
    """Manages theme switching and provides consistent styling"""
    
    def __init__(self, theme_name: str = "Dark"):
        self.theme_name = theme_name
        self.theme = ThemeConfig.DARK if theme_name == "Dark" else ThemeConfig.LIGHT
        
    def get_color(self, color_name: str) -> str:
        """Get color value from current theme"""
        return self.theme.get(color_name, self.theme['primary'])
    
    def get_chart_colors(self, count: int) -> list:
        """Get chart colors, cycling through the palette if needed"""
        colors = self.theme['chart_colors']
        if count <= len(colors):
            return colors[:count]
        # Cycle through colors if we need more
        return [colors[i % len(colors)] for i in range(count)]
    
    def apply_base_layout(self, fig: go.Figure, title: str = "", height: int = 600) -> go.Figure:
        """Apply consistent base styling to any figure"""
        fig.update_layout(
                template='plotly_dark' if self.theme_name == "Dark" else "plotly_white",
                plot_bgcolor=self.theme['plot_bgcolor'],
                paper_bgcolor=self.theme['paper_bgcolor'],
                font=dict(
                    family='Inter, Montserrat, Arial, sans-serif',
                    size=14,
                    color=self.theme['text_color']
                ),
            margin=dict(l=60, r=60, t=80, b=60),
                title=dict(
                    text=title,
                    font=dict(
                        size=24,
                        color=self.theme['text_color'],
                        family='Inter, Montserrat, Arial, sans-serif'
                    ),
                    x=0.5,
                    xanchor='center',
                    pad=dict(t=20)
                ),
            legend=dict(
                    bgcolor=f"rgba({self._hex_to_rgb(self.theme['bg_color'])}, 0.9)",
                    bordercolor=self.theme['border_color'],
                borderwidth=1,
                    font=dict(color=self.theme['text_color'], size=12),
                orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=self.theme['grid_color'],
                    zerolinecolor=self.theme['grid_color_secondary'],
                    linecolor=self.theme['border_color'],
                    tickfont=dict(color=self.theme['text_color'], size=12),
                    title_font=dict(size=14, color=self.theme['text_color'])
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=self.theme['grid_color'],
                    zerolinecolor=self.theme['grid_color_secondary'],
                    linecolor=self.theme['border_color'],
                    tickfont=dict(color=self.theme['text_color'], size=12),
                    title_font=dict(size=14, color=self.theme['text_color'])
                ),
            hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=self.theme['hover_bg'],
                    bordercolor=self.theme['hover_border'],
                    font=dict(color=self.theme['text_color'])
                ),
                height=height
        )
        return fig

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for transparency"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"{rgb[0]}, {rgb[1]}, {rgb[2]}"

# =============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# =============================================================================

class AdvancedVisualizer:
    """Advanced visualization class with modern styling and features"""
    
    def __init__(self, theme_name: str = "Dark"):
        self.theme_manager = ThemeManager(theme_name)
        
    def create_advanced_line_chart(self, df: pd.DataFrame, title: str = "Advanced Line Chart", period: Optional[str] = None) -> go.Figure:
        """Create a beautiful, advanced line chart with multiple styling options"""
        fig = go.Figure()
        # Get colors for each column
        colors = self.theme_manager.get_chart_colors(len([c for c in df.columns if c != 'Date']))
        
        for idx, col in enumerate([c for c in df.columns if c != 'Date']):
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[col],
                name=col,
                line=dict(
                    color=colors[idx],
                    width=3,
                    shape='spline'  # Smooth curves
                ),
                mode='lines',
                hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>',
                fill=None  # Disable fill under line
            ))
        
        # Apply advanced styling
        fig = self.theme_manager.apply_base_layout(fig, title)
        
        # Add range slider with theme colors
        fig.update_xaxes(
            rangeslider=dict(
                visible=True,
                bgcolor=self.theme_manager.get_color('bg_color'),
                bordercolor=self.theme_manager.get_color('border_color'),
                borderwidth=1
            )
        )
        
        # Ensure y-axis is auto scaling (default behavior, so no fixed range)
        fig.update_yaxes(autorange=True)
        
        return fig

    
    def create_candlestick_chart(self, df: pd.DataFrame, period: Optional[str] = None, 
                                title: str = "Advanced Candlestick Chart") -> go.Figure:
        """Create an advanced candlestick chart with technical indicators"""
        if period:
            df = fn.filter_data(df, period)
        
        fig = go.Figure()
        
        # Main candlestick
        fig.add_trace(go.Candlestick(
            x=df['Date'], 
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=self.theme_manager.get_color('success'),
            decreasing_line_color=self.theme_manager.get_color('danger'),
            increasing_fillcolor=f"rgba({self.theme_manager._hex_to_rgb(self.theme_manager.get_color('success'))}, 0.3)",
            decreasing_fillcolor=f"rgba({self.theme_manager._hex_to_rgb(self.theme_manager.get_color('danger'))}, 0.3)",
            name="Price"
        ))
        
        # Add moving averages
        if len(df) > 50:
            df['SMA_20'] = pta.sma(df['Close'], 20)
            df['SMA_50'] = pta.sma(df['Close'], 50)
            
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['SMA_20'],
                name="SMA 20",
                line=dict(color=self.theme_manager.get_color('info'), width=2, dash='dash'),
                hovertemplate='<b>SMA 20</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['SMA_50'],
                name="SMA 50",
                line=dict(color=self.theme_manager.get_color('accent'), width=2, dash='dot'),
                hovertemplate='<b>SMA 50</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
        
        fig = self.theme_manager.apply_base_layout(fig, title, height=700)
        
        # Add range slider
        fig.update_xaxes(
            rangeslider=dict(
                visible=True,
                bgcolor=self.theme_manager.get_color('bg_color'),
                bordercolor=self.theme_manager.get_color('border_color'),
                borderwidth=1
            )
        )
        
        return fig

    def create_technical_indicators(self, df: pd.DataFrame, indicators: list = None) -> Dict[str, go.Figure]:
        """Create multiple technical indicator charts"""
        if indicators is None:
            indicators = ['RSI', 'MACD', 'Volume']
        
        charts = {}
        
        if 'RSI' in indicators:
            charts['RSI'] = self._create_rsi_chart(df)
        
        if 'MACD' in indicators:
            charts['MACD'] = self._create_macd_chart(df)
        
        if 'Volume' in indicators:
            charts['Volume'] = self._create_volume_chart(df)
        
        return charts
    
    def _create_rsi_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create RSI chart with overbought/oversold zones"""
        df_copy = df.copy()
        df_copy['RSI'] = pta.rsi(df_copy['Close'])
        
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(go.Scatter(
            x=df_copy['Date'],
            y=df_copy['RSI'],
            name='RSI',
            line=dict(color=self.theme_manager.get_color('primary'), width=3),
            fill='tonexty',
            fillcolor=f"rgba({self.theme_manager._hex_to_rgb(self.theme_manager.get_color('primary'))}, 0.1)"
        ))
        
        # Overbought line
        fig.add_trace(go.Scatter(
            x=df_copy['Date'],
            y=[70] * len(df_copy),
            name="Overbought (70)",
            line=dict(color=self.theme_manager.get_color('danger'), width=2, dash="dash"),
            hovertemplate='<b>Overbought Level</b><br>Value: 70<extra></extra>'
        ))
        
        # Oversold line
        fig.add_trace(go.Scatter(
            x=df_copy['Date'],
            y=[30] * len(df_copy),
            name="Oversold (30)",
            line=dict(color=self.theme_manager.get_color('warning'), width=2, dash="dash"),
            fill="tonexty",
            fillcolor=f"rgba({self.theme_manager._hex_to_rgb(self.theme_manager.get_color('warning'))}, 0.1)"
        ))
        
        fig = self.theme_manager.apply_base_layout(fig, "📈 RSI (Relative Strength Index)", height=300)
        fig.update_yaxes(range=[0, 100])
        
        return fig
    
    def _create_macd_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create MACD chart with signal line and histogram"""
        df_copy = df.copy()
        macd_data = pta.macd(df_copy['Close'])
        df_copy["MACD"] = macd_data.iloc[:, 0]
        df_copy["MACD_Signal"] = macd_data.iloc[:, 1]
        df_copy["MACD_Hist"] = macd_data.iloc[:, 2]
        
        fig = go.Figure()
        
        # MACD line
        fig.add_trace(go.Scatter(
            x=df_copy['Date'],
            y=df_copy['MACD'],
            name="MACD",
            line=dict(color=self.theme_manager.get_color('primary'), width=3),
            hovertemplate="<b>MACD</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>"
        ))
        
        # Signal line
        fig.add_trace(go.Scatter(
            x=df_copy['Date'],
            y=df_copy["MACD_Signal"],
            name="Signal",
            line=dict(color=self.theme_manager.get_color('secondary'), width=2, dash="dash"),
            hovertemplate="<b>Signal</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>"
        ))
        
        # Histogram
        colors = ['#00ff88' if val > 0 else '#ff4757' for val in df_copy["MACD_Hist"]]
        fig.add_trace(go.Bar(
            x=df_copy['Date'],
            y=df_copy["MACD_Hist"],
            name="Histogram",
            marker_color=colors,
            opacity=0.6,
            hovertemplate="<b>Histogram</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>"
        ))
        
        fig = self.theme_manager.apply_base_layout(fig, "MACD (Moving Average Convergence Divergence)", height=300)
        
        return fig
    
    def _create_volume_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create volume chart with price correlation"""
        fig = go.Figure()
        
        # Volume bars
        colors = ['#00ff88' if close > open else '#ff4757' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
            hovertemplate="<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>"
        ))
        
        fig = self.theme_manager.apply_base_layout(fig, "📊 Trading Volume", height=300)
        
        return fig
    
    def create_advanced_table(self, df: pd.DataFrame, title: str = "Data Table") -> go.Figure:
         """Create a beautiful, interactive table"""
         header_values = [''] + [f'{str(col)[:20]}' for col in df.columns]
         cell_values = [df.index.astype(str).tolist()] + [df[col].astype(str).tolist() for col in df.columns]
        
         num_rows = len(df)
         row_even = self.theme_manager.get_color('table_row_even')
         row_odd = self.theme_manager.get_color('table_row_odd')
    
         fill_colors = []
         for _ in cell_values:
            fill_colors.append([row_odd if i % 2 else row_even for i in range(num_rows)])

         fig = go.Figure(go.Table(
            header=dict(
                values=header_values,
                    fill_color=self.theme_manager.get_color('table_header'),
                align='center',
                font=dict(color="white", size=20, family='Inter, Montserrat, Arial, sans-serif'),
            ),
            cells=dict(
                values=cell_values,
                fill_color=fill_colors,
                align='left',
                    font=dict(color=self.theme_manager.get_color('text_color'), size=18),
                    line_color=self.theme_manager.get_color('border_color')
            )
        ))
            
         fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor=self.theme_manager.get_color('paper_bgcolor'),
        )
            
         return fig

# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

# Initialize default theme manager for backward compatibility
_default_visualizer = AdvancedVisualizer("Dark")

@st.cache_data
def interactive_plot(df: pd.DataFrame) -> go.Figure:
    """Legacy function - creates interactive line plot"""
    return _default_visualizer.create_advanced_line_chart(df, title="Interactive Price Chart")

@st.cache_data
def plotly_table(df: pd.DataFrame) -> go.Figure:
    """Legacy function - creates beautiful table"""
    return _default_visualizer.create_advanced_table(df)

@st.cache_data
def close_chart(df: pd.DataFrame,period: Optional[str] = None) -> go.Figure:
    """Legacy function - creates OHLC chart"""
    if period:
        df=fn.filter_data(df,period)
        
    selected_columns = ['Date','Open', 'Close', 'High', 'Low']  # list your specific columns here
    new_df = df[selected_columns]
    return _default_visualizer.create_advanced_line_chart(new_df, title="Stock Price Chart (OHLC)")

@st.cache_data
def candlestick(df: pd.DataFrame, period: Optional[str] = None) -> go.Figure:
    """Legacy function - creates candlestick chart"""
    return _default_visualizer.create_candlestick_chart(df, period)

@st.cache_data
def RSI(df: pd.DataFrame, period: Optional[str] = None) -> go.Figure:
    """Legacy function - creates RSI chart"""
    if period:
        df = fn.filter_data(df, period)
    return _default_visualizer._create_rsi_chart(df)

@st.cache_data
def Moving_average(df: pd.DataFrame, period: Optional[str] = None) -> go.Figure:
    """Legacy function - creates moving average chart"""
    if period:
        df = fn.filter_data(df, period)    
    selected_columns = ['Date','Open', 'Close', 'High', 'Low']  # list your specific columns here
    new_df = df[selected_columns]
    return _default_visualizer.create_advanced_line_chart(new_df, title="Stock Prices with Moving Average")

@st.cache_data
def MACD(df: pd.DataFrame, period: Optional[str] = None) -> go.Figure:
    """Legacy function - creates MACD chart"""
    if period:
        df = fn.filter_data(df, period)
    return _default_visualizer._create_macd_chart(df)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def download_chart(fig: go.Figure, filename: str = 'chart.png') -> None:
    """Download Plotly chart as PNG"""
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    st.download_button(
        label=" Download Chart PNG",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )



def get_theme_manager(theme_name: str = "Dark") -> ThemeManager:
    """Get theme manager instance"""
    return ThemeManager(theme_name)

def get_visualizer(theme_name: str = "Dark") -> AdvancedVisualizer:
    """Get advanced visualizer instance"""
    return AdvancedVisualizer(theme_name)

# =============================================================================
# THEME SWITCHER
# =============================================================================

def create_theme_switcher() -> str:
    """Create a theme switcher in the sidebar"""
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        theme = st.selectbox(
            "Choose Theme",
            ["Dark", "Light"],
            index=0,
            key="theme_selector"
        )
        
        if theme == "Dark":
            st.markdown("🌙 Dark theme active")
        else:
            st.markdown("☀️ Light theme active")
    
    return theme

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the new visualization system"""
    st.markdown("""
    ## 🚀 Advanced Visualization System Usage
    
    ### 1. Get a visualizer instance:
    ```python
    visualizer = get_visualizer("Dark")  # or "Light"
    ```
    
    ### 2. Create advanced charts:
    ```python
    # Advanced line chart
    fig = visualizer.create_advanced_line_chart(df, "My Chart")
    
    # Candlestick with indicators
    fig = visualizer.create_candlestick_chart(df, "1y", "Stock Analysis")
    
    # Technical indicators
    charts = visualizer.create_technical_indicators(df, ['RSI', 'MACD'])
    ```
    
    ### 3. Change theme globally:
    ```python
    # In your main app, call this to get current theme
    current_theme = create_theme_switcher()
    visualizer = get_visualizer(current_theme)
    ```
    
    ### 4. Customize colors in ThemeConfig:
    - Edit the DARK and LIGHT dictionaries in ThemeConfig class
    - All charts will automatically use the new colors
    """)

# Initialize theme switcher when module is imported
if 'theme_selector' not in st.session_state:
    st.session_state.theme_selector = "Dark"
