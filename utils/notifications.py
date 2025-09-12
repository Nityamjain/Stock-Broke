import streamlit as st
import time
from datetime import datetime
import pandas as pd
import uuid

def show_floating_notification(position: str = 'bottom-right', duration_sec: int = 8):
    """Show a floating notification that doesn't take up page space.

    position: 'bottom-right' | 'top-right' | 'bottom-left' | 'top-left'
    duration_sec: seconds to keep the popup visible before auto-dismiss
    """
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []

    # Global CSS for animations (defined once)
    st.markdown("""
    <style>
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Remove expired notifications (batch processing)
    current_time = time.time()
    st.session_state.notifications = [
        n for n in st.session_state.notifications
        if current_time - n.get('timestamp', 0) <= max(1, duration_sec)
    ]

    if not st.session_state.notifications:
        return

    # Show latest notification
    latest = st.session_state.notifications[-1]
    notification_key = f"notification_{id(latest)}"  # Unique key for dismiss button

    # Positioning
    pos_styles = {
        'bottom-right': 'bottom: 20px; right: 20px;',
        'top-right': 'top: 20px; right: 20px;',
        'bottom-left': 'bottom: 20px; left: 20px;',
        'top-left': 'top: 20px; left: 20px;'
    }
    anchor_style = pos_styles.get(position, 'bottom: 20px; right: 20px;')

    notification_html = f"""
    <div style="
        position: fixed;
        {anchor_style}
        z-index: 1000;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 350px;
        font-family: 'Source Sans Pro', sans-serif;
        animation: slideIn 0.3s ease-out;
    ">
    """

    if latest['type'] == 'success':
        notification_html += f"""
        <div style="
            background: linear-gradient(135deg, #00d4aa, #00b894);
            color: white;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #00a085;
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">🎉 {latest['title']}</div>
            <div style="font-size: 14px; opacity: 0.9;">{latest['message']}</div>
        </div>
        """
    elif latest['type'] == 'error':
        notification_html += f"""
        <div style="
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #c44569;
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">❌ {latest['title']}</div>
            <div style="font-size: 14px; opacity: 0.9;">{latest['message']}</div>
        </div>
        """
    elif latest['type'] == 'warning':
        notification_html += f"""
        <div style="
            background: linear-gradient(135deg, #feca57, #ff9ff3);
            color: #2c3e50;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #f39c12;
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">⚠️ {latest['title']}</div>
            <div style="font-size: 14px; opacity: 0.9;">{latest['message']}</div>
        </div>
        """
    elif latest['type'] == 'info':
        notification_html += f"""
        <div style="
            background: linear-gradient(135deg, #48dbfb, #0abde3);
            color: white;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #54a0ff;
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">ℹ️ {latest['title']}</div>
            <div style="font-size: 14px; opacity: 0.9;">{latest['message']}</div>
        </div>
        """

    notification_html += "</div>"
    st.markdown(notification_html, unsafe_allow_html=True)

    # Streamlit-based dismiss button (optional)
    try:
        if st.button("Dismiss", key=notification_key):
            if st.session_state.notifications:
                st.session_state.notifications.pop()
                st.rerun()
    except Exception:
        pass

def show_notification_popup(notification_type: str, title: str, message: str,
                             position: str = 'bottom-right', duration_sec: int = 8):
    """Convenience helper: enqueue a notification and render a popup immediately."""
    add_notification(notification_type=notification_type, title=title, message=message)
    show_floating_notification(position=position, duration_sec=duration_sec)

def show_training_status_sidebar():
    """Show training status in the sidebar."""
    if 'background_trainings' not in st.session_state:
        st.session_state.background_trainings = {}
        st.sidebar.write("DEBUG: Initialized empty background_trainings")
        return

    active_trainings = []
    completed_trainings = []
    failed_trainings = []

    for training_id, config in st.session_state.background_trainings.items():
        if not isinstance(training_id, str):
            st.sidebar.warning(f"Invalid training_id: {training_id}")
            continue
        if config.get('status') == 'running':
            active_trainings.append((training_id, config))
        elif config.get('status') == 'completed':
            completed_trainings.append((training_id, config))
        elif config.get('status') == 'failed':
            failed_trainings.append((training_id, config))

    if not (active_trainings or completed_trainings or failed_trainings):
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Training Status")
    st.sidebar.write("DEBUG: Rendering training status")

    # Active trainings
    if active_trainings:
        st.sidebar.markdown("**🔄 Active Trainings:**")
        for training_id, config in active_trainings:
            try:
                start_time = datetime.fromisoformat(config.get('start_time', datetime.now().isoformat()))
                duration = datetime.now() - start_time
                with st.sidebar.expander(f"🔄 {config.get('stock_code', 'Training')}", expanded=False):
                    st.write(f"**ID:** {training_id}")
                    st.write(f"**Duration:** {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60:.0f}s")
                    st.write(f"**Market:** {config.get('market', 'Unknown')}")
                    if st.button("View Details", key=f"sidebar_view_{training_id}_{uuid.uuid4()}"):
                        st.session_state.selected_training = training_id
                        st.rerun()
            except ValueError as e:
                st.sidebar.warning(f"Invalid start_time for training {training_id}: {str(e)}")

    # Completed trainings
    if completed_trainings:
        st.sidebar.markdown("**✅ Completed Trainings:**")
        for training_id, config in completed_trainings:  # Removed arbitrary limit
            try:
                completion_time = datetime.fromisoformat(config.get('completion_time', datetime.now().isoformat()))
                with st.sidebar.expander(f"✅ {config.get('stock_code', 'Training')}", expanded=False):
                    st.write(f"**Completed:** {completion_time.strftime('%H:%M')}")
                    if st.button("View Results", key=f"sidebar_results_{training_id}_{uuid.uuid4()}"):
                        st.session_state.selected_training = training_id
                        st.rerun()
            except ValueError as e:
                st.sidebar.warning(f"Invalid completion_time for training {training_id}: {str(e)}")

    # Failed trainings
    if failed_trainings:
        st.sidebar.markdown("**❌ Failed Trainings:**")
        for training_id, config in failed_trainings:  # Removed arbitrary limit
            with st.sidebar.expander(f"❌ {config.get('stock_code', 'Training')}", expanded=False):
                st.write(f"**Error:** {config.get('error', 'Unknown')[:50]}...")
                if st.button("View Details", key=f"sidebar_error_{training_id}_{uuid.uuid4()}"):
                    st.session_state.selected_training = training_id
                    st.rerun()

    # Quick actions
    if completed_trainings or failed_trainings:
        st.sidebar.markdown("---")
        if st.sidebar.button("🧹 Clear Old Records", key=f"sidebar_clear_{uuid.uuid4()}"):
            try:
                from utils.background_training import BackgroundTrainingManager
                if 'training_manager' in st.session_state:
                    st.session_state.training_manager.clear_completed_trainings()
                    st.session_state.background_trainings = {
                        k: v for k, v in st.session_state.background_trainings.items()
                        if v.get('status') == 'running'
                    }
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to clear records: {str(e)}")

def show_training_details_page(training_id: str):
    """Show training details on a dedicated page view."""
    if not training_id or 'background_trainings' not in st.session_state:
        st.error("No training records found or invalid training ID.")
        return

    config = st.session_state.background_trainings.get(training_id)
    if not config:
        st.error(f"Training {training_id} not found.")
        return

    # Back button
    if st.button("← Back to Main Page", key=f"back_main_{training_id}_{uuid.uuid4()}"):
        st.session_state.selected_training = None
        st.rerun()

    st.subheader(f"📊 Training Details: {training_id}")
    st.write("DEBUG: Rendering training details page")

    # Basic info
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Status:** {config.get('status', 'Unknown')}")
            start_time = config.get('start_time', 'Unknown')
            if start_time != 'Unknown':
                try:
                    start_time = datetime.fromisoformat(start_time).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    start_time = 'Invalid format'
            st.write(f"**Start Time:** {start_time}")
            completion_time = config.get('completion_time')
            if completion_time:
                try:
                    completion_time = datetime.fromisoformat(completion_time).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    completion_time = 'Invalid format'
                st.write(f"**Completion Time:** {completion_time}")

        with col2:
            st.write(f"**Lookback:** {config.get('lookback', 'Unknown')}")
            st.write(f"**Batch Size:** {config.get('batch_size', 'Unknown')}")
            st.write(f"**Epochs:** {config.get('num_epochs', 'Unknown')}")
    except Exception as e:
        st.error(f"Failed to display basic info: {str(e)}")

    # Parameters
    st.markdown("## ⚙️ Model Parameters")
    try:
        params = {
            'Learning Rate': config.get('learning_rate', 'Unknown'),
            'Hidden Size': config.get('hidden_size', 'Unknown'),
            'LSTM Layers': config.get('num_layers', 'Unknown'),
            'Dropout': config.get('dropout', 'Unknown'),
            'Market': config.get('market', 'Unknown'),
            'Stock': config.get('stock_code', 'Unknown')
        }
        for param, value in params.items():
            st.write(f"**{param}:** {value}")
    except Exception as e:
        st.error(f"Failed to display parameters: {str(e)}")

    # Results (if completed)
    if config.get('status') == 'completed' and config.get('results'):
        st.markdown("## 📈 Training Results")
        try:
            results = config.get('results', {})
            if 'metrics' in results:
                metrics = results['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train MAE", f"${metrics.get('train_mae', 0):.2f}")
                with col2:
                    st.metric("Train RMSE", f"${metrics.get('train_rmse', 0):.2f}")
                with col3:
                    st.metric("Test MAE", f"${metrics.get('test_mae', 0):.2f}")
                with col4:
                    st.metric("Test RMSE", f"${metrics.get('test_rmse', 0):.2f}")
        except Exception as e:
            st.error(f"Failed to display metrics: {str(e)}")

        if 'future_predictions' in results:
            st.markdown("## 🔮 Future Predictions")
            try:
                future_data = results['future_predictions']
                # Handle different data formats
                if isinstance(future_data, (list, dict)):
                    future_df = pd.DataFrame(future_data)
                    st.dataframe(future_df, use_container_width=True)
                elif isinstance(future_data, pd.DataFrame):
                    st.dataframe(future_data, use_container_width=True)
                elif isinstance(future_data, str):
                    try:
                        future_df = pd.read_json(future_data)
                        st.dataframe(future_df, use_container_width=True)
                    except ValueError:
                        st.write("Future predictions data available but not in a displayable format")
                        st.write("Raw data:", future_data)
                else:
                    st.warning("Future predictions data not in a displayable format")
                    st.write("Raw data:", future_data)
            except Exception as e:
                st.error(f"Could not display future predictions: {str(e)}")
                st.write("Raw data:", future_data)

    # Error details (if failed)
    if config.get('status') == 'failed':
        st.markdown("## ❌ Error Details")
        st.error(config.get('error', 'Unknown error occurred'))

    # Actions
    try:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Main Page", key=f"back_main_{training_id}_{uuid.uuid4()}"):
                st.session_state.selected_training = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear This Record", key=f"clear_{training_id}_{uuid.uuid4()}"):
                if training_id in st.session_state.background_trainings:
                    del st.session_state.background_trainings[training_id]
                    st.session_state.selected_training = None
                    st.rerun()
    except Exception as e:
        st.error(f"Failed to render action buttons: {str(e)}")

def add_notification(notification_type: str, title: str, message: str, training_id: str = None):
    """Add a new notification to the queue."""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []

    if notification_type not in ['success', 'error', 'warning', 'info']:
        notification_type = 'info'  # Default to info if invalid type

    notification = {
        'type': notification_type,
        'title': title,
        'message': message,
        'training_id': training_id,
        'timestamp': datetime.now().timestamp()
    }

    st.session_state.notifications.append(notification)

    # Keep only last 3 notifications
    if len(st.session_state.notifications) > 3:
        st.session_state.notifications.pop(0)

    # Uncomment for debugging
    # st.write(f"DEBUG: Added notification - {title}: {message}")