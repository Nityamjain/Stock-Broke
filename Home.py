import streamlit as st
import pkg_resources
import utils.vizuals as vz



try:
    streamlit_version = pkg_resources.get_distribution("streamlit").version
    if pkg_resources.parse_version(streamlit_version) < pkg_resources.parse_version("0.86.0"):
        st.warning("⚠️ Streamlit version is older than 0.86.0. Some features may not work. Please upgrade Streamlit.")
except pkg_resources.DistributionNotFound:
    st.warning("⚠️ Unable to verify Streamlit version. Ensure Streamlit is installed correctly.")

try:
    from utils.functions import render_feedback_form
    from utils.notifications import show_floating_notification, show_training_status_sidebar
    from utils.background_training import BackgroundTrainingManager
except ImportError as e:
    st.error(f"❌ Failed to import custom utils: {str(e)}. Please check the utils module.")
    st.stop()

# Page configuration
try:
    st.set_page_config(
        page_title="Stock Broke",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"❌ Failed to set page configuration: {str(e)}")
    st.stop()

    

# Custom CSS for modern styling
st.markdown("""
<style>
.hero {
  padding: 48px;
  border-radius: 16px;
  margin-bottom: 24px;
  background: radial-gradient(1200px 600px at 0% 0%, #2b2f55 0%, #0d0f1a 60%);
  color: #fff;
}
.hero h1 { font-size: 40px; margin: 0 0 8px 0; letter-spacing: 0.5px; }
.hero p { color: #d0d3e2; margin: 0; }

.card { background: #10121f; border: 1px solid #1a1e34; border-radius: 12px; padding: 18px; }
.card h3 { color: #e6e8f0; margin: 0 0 8px 0; font-weight: 600; }
.card p, .card li { color: #aab0c0; }

.cta { display: inline-block; padding: 8px 14px; border-radius: 8px; border: 1px solid #3140a0; color: #e6e8f0; text-decoration: none; }
.cta:hover { background: #19215c; }

.subtle { color: #9aa1b4; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function to render the Stock Analysis Platform home page."""
    # Initialize background training manager
    if 'training_manager' not in st.session_state:
        try:
            st.session_state.training_manager = BackgroundTrainingManager()
        except Exception as e:
            st.error(f"❌ Failed to initialize training manager: {str(e)}")
            return
    try:
        st.markdown("""
        <div class="hero">
          <h1>STOCK BROKE : Stock Analysis Platform</h1>
          <p>Research. Predict. Monitor. One workspace for markets.</p>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns([1,1])
        with cols[0]:
            if st.button('Login / Sign Up'):
                st.switch_page('pages/Login.py')
    except Exception as e:
        st.error(f"❌ Failed to render header: {str(e)}")

    # Key metrics
    st.markdown("## Platform Overview")
    try:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.markdown("""
            <div class="card">
              <h3>Predictive Models</h3>
              <p>LSTM-based forecasts with robust validation and early-stopping.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
              <h3>Technical Charts</h3>
              <p>Interactive RSI, MACD, moving averages, and price studies.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="card">
              <h3>Risk & CAPM</h3>
              <p>Estimate beta, expected return, and compare against benchmarks.</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="card">
              <h3>Global Markets</h3>
              <p>NSE, NASDAQ, and more—unified tickers and data retrieval.</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Failed to render Platform Overview: {str(e)}")

    # Key Features section with debugging
    st.markdown("## Key Features")
    try:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div class="card">
            <h3>Advanced Stock Prediction</h3>
            <ul>
                <li>LSTM neural networks with attention mechanism</li>
                <li>Market-specific parameter optimization</li>
                <li>30-day future price forecasts</li>
                <li>Background training with notifications</li>
                <li>CSV download for all predictions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(" ")
            st.markdown("""
            <div class="card">
            <h3>Technical Analysis</h3>
            <ul>
                <li>RSI, MACD, Moving Averages</li>
                <li>Volume analysis and patterns</li>
                <li>Interactive charts and indicators</li>
                <li>Real-time market data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
            <h3>CAPM Analysis</h3>
            <ul>
                <li>Risk-return calculations</li>
                <li>Beta coefficient analysis</li>
                <li>Portfolio optimization tools</li>
                <li>Market risk assessment</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(" ")
            st.markdown("""
            <div class="card">
            <h3>Multi-Market Support</h3>
            <ul>
                <li>S&P 500, NASDAQ, Dow Jones</li>
                <li>Nifty 50, FTSE 100</li>
                <li>Market-specific optimizations</li>
                <li>Global stock coverage</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Failed to render Key Features section: {str(e)}")
        st.write("DEBUG: Check CSS (.feature-card) and HTML syntax.")

    # Navigation guide
    st.markdown("## Navigation")
    try:
        st.info("""
        Use the sidebar to access:
        - Home (overview)
        - Stock Prediction (training + forecast)
        - Stock Analysis (indicators & charts)
        - CAPM (risk-return)
        - Watchlist (save & monitor)
        - Login (sign in/up)
        """)
    except Exception as e:
        st.error(f"❌ Failed to render Navigation Guide: {str(e)}")

    # Background training status
    try:
        if hasattr(st.session_state.training_manager, 'is_training_active') and st.session_state.training_manager.is_training_active():
            st.markdown("## Active Background Training")
            st.success("You have an active training running. Check the status in the sidebar or the Stock Prediction page.")
    except AttributeError:
        st.warning("⚠️ Training status check unavailable. Please verify BackgroundTrainingManager implementation.")
    except Exception as e:
        st.error(f"❌ Error checking training status: {str(e)}")

    # Render feedback form
    try:
        render_feedback_form()
    except Exception as e:
        st.warning(f"⚠️ Feedback form unavailable: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application failed to start: {str(e)}")
        st.write("DEBUG: Check Streamlit version, custom utils, and browser console for errors.")