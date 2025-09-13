import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json

# --- Firebase Init ---
if not firebase_admin._apps:
    service_account_str = st.secrets["FIREBASE"]["json"]
    service_account_info = json.loads(service_account_str)
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- Authentication Functions ---
def login(email: str):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        return True
    except:
        return False

def signup(email: str, password: str, username: str):
    try:
        user = auth.create_user(
            email=email,
            password=password,
            uid=username
        )
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        return True
    except Exception as e:
        st.error(f"Error creating account: {str(e)}")
        return False

def signout():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ''
    st.session_state.usermail = ''
    st.session_state.pop('watchlist', None)
    st.query_params.clear()
    st.rerun()
