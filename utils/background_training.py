import streamlit as st
import threading
import time
import queue
import json
from datetime import datetime
import os
from typing import Dict, Any, Optional
import pickle

class BackgroundTrainingManager:
    """Manages background training processes with notifications"""
    
    def __init__(self):
        self.training_queue = queue.Queue()
        self.results_cache = {}
        self.notification_queue = queue.Queue()
        
    def start_background_training(self, training_config: Dict[str, Any], training_function):
        """Start training in background thread"""
        if self.is_training_active():
            st.warning("⚠️ Training already in progress. Please wait for completion.")
            return False
        
        # Create unique training ID
        training_id = f"training_{int(time.time())}"
        
        # Create a clean config for storage (remove non-serializable items)
        clean_config = self._create_clean_config(training_config)
        
        # Store training config
        clean_config['training_id'] = training_id
        clean_config['start_time'] = datetime.now().isoformat()
        clean_config['status'] = 'running'
        
        # Save to session state
        if 'background_trainings' not in st.session_state:
            st.session_state.background_trainings = {}
        
        st.session_state.background_trainings[training_id] = clean_config
        
        # Start background thread with original config (including DataFrame)
        training_thread = threading.Thread(
            target=self._run_training_background,
            args=(training_id, training_config, training_function),
            daemon=True
        )
        training_thread.start()
        
        st.success(f"🚀 Background training started! Training ID: {training_id}")
        st.info("💡 You can navigate to other pages while training continues in the background.")
        
        return training_id
    
    def _create_clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a clean, serializable version of the config for storage"""
        clean_config = {}
        
        for key, value in config.items():
            if key == 'df':
                # Store DataFrame info instead of the actual DataFrame
                if hasattr(value, '__len__'):
                    clean_config[key] = f"DataFrame with {len(value)} rows"
                else:
                    clean_config[key] = "DataFrame"
            elif hasattr(value, 'isoformat'):
                # Convert datetime objects to strings
                clean_config[key] = str(value)
            elif isinstance(value, (str, int, float, bool, list, dict)):
                # Keep serializable types as-is
                clean_config[key] = value
            else:
                # Convert other types to string representation
                clean_config[key] = str(value)
        
        return clean_config
    
    def _run_training_background(self, training_id: str, config: Dict[str, Any], training_function):
        """Run training in background thread"""
        try:
            # Update status
            config['status'] = 'running'
            self._update_training_status(training_id, config)
            
            # Run training function
            results = training_function(config)
            
            # Clean results for storage
            clean_results = self._clean_results_for_storage(results)
            
            # Store results
            config['status'] = 'completed'
            config['completion_time'] = datetime.now().isoformat()
            config['results'] = clean_results
            
            # Save results to cache
            self.results_cache[training_id] = results  # Keep original results in cache
            
            # Save to session state
            self._update_training_status(training_id, config)
            
            # Add to notification queue
            self.notification_queue.put({
                'type': 'success',
                'title': 'Training Completed! 🎉',
                'message': f'Training {training_id} has completed successfully.',
                'training_id': training_id
            })
            
        except Exception as e:
            # Handle errors
            config['status'] = 'failed'
            config['error'] = str(e)
            config['completion_time'] = datetime.now().isoformat()
            
            self._update_training_status(training_id, config)
            
            # Add error notification
            self.notification_queue.put({
                'type': 'error',
                'title': 'Training Failed! ❌',
                'message': f'Training {training_id} failed: {str(e)}',
                'training_id': training_id
            })
    
    def _update_training_status(self, training_id: str, config: Dict[str, Any]):
        """Update training status in session state"""
        if 'background_trainings' in st.session_state:
            st.session_state.background_trainings[training_id] = config
    
    def is_training_active(self) -> bool:
        """Check if any training is currently active"""
        if 'background_trainings' not in st.session_state:
            return False
        
        for training_id, config in st.session_state.background_trainings.items():
            if config.get('status') == 'running':
                return True
        return False
    
    def get_active_training(self) -> Optional[Dict[str, Any]]:
        """Get currently active training if any"""
        if 'background_trainings' not in st.session_state:
            return None
        
        for training_id, config in st.session_state.background_trainings.items():
            if config.get('status') == 'running':
                return {'training_id': training_id, **config}
        return None
    
    def get_training_results(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get training results by ID"""
        # Check cache first
        if training_id in self.results_cache:
            return self.results_cache[training_id]
        
        # Check session state
        if 'background_trainings' in st.session_state:
            config = st.session_state.background_trainings.get(training_id)
            if config and config.get('status') == 'completed':
                return config.get('results')
        
        return None
    
    def get_all_trainings(self) -> Dict[str, Any]:
        """Get all training records"""
        return st.session_state.get('background_trainings', {})
    
    def clear_completed_trainings(self):
        """Clear completed training records to free memory"""
        if 'background_trainings' in st.session_state:
            completed_ids = []
            for training_id, config in st.session_state.background_trainings.items():
                if config.get('status') in ['completed', 'failed']:
                    completed_ids.append(training_id)
            
            for training_id in completed_ids:
                del st.session_state.background_trainings[training_id]
                if training_id in self.results_cache:
                    del self.results_cache[training_id]
    
    def get_notifications(self) -> list:
        """Get all pending notifications"""
        notifications = []
        while not self.notification_queue.empty():
            try:
                notification = self.notification_queue.get_nowait()
                notifications.append(notification)
            except queue.Empty:
                break
        return notifications

    def _clean_results_for_storage(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean results for storage in session state"""
        clean_results = {}
        
        for key, value in results.items():
            if key == 'future_predictions':
                # Handle DataFrame specially
                if hasattr(value, 'to_dict'):
                    clean_results[key] = value.to_dict('records')
                else:
                    clean_results[key] = str(value)
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                clean_results[key] = self._clean_results_for_storage(value)
            elif isinstance(value, (str, int, float, bool, list)):
                # Keep serializable types as-is
                clean_results[key] = value
            else:
                # Convert other types to string representation
                clean_results[key] = str(value)
        
        return clean_results
