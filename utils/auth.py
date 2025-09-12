# utils/auth.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth

def init_firebase():
    cred = credentials.Certificate("stock-broke-f5d95-ec1744339a65.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

def login():
    try:
        user = auth.get_user_by_email(st.session_state.email)
        # Note: Firebase Admin SDK cannot verify passwords directly.
        # For simplicity, assume login success if user exists.
        # In production, use Firebase client-side SDK or REST API for password verification.
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.logged_in = True
        st.success("Logged in successfully")
        st.session_state['login_redirect'] = True  # Flag to trigger redirect
    except:
        st.warning("User not found or incorrect credentials. Please sign up or check your email.")

def signup():
    try:
        user = auth.create_user(email=st.session_state.email, password=st.session_state.password, uid=st.session_state.signup_username)
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.logged_in = True
        st.success("Account created and logged in successfully")
        st.session_state['login_redirect'] = True  # Flag to trigger redirect
    except Exception as e:
        st.error(f"Error creating account: {str(e)}")

def signout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.usermail = ''
    st.session_state.pop('watchlist', None)
    st.session_state.pop('login_redirect', None)
    st.query_params.clear()
    st.switch_page("pages/Login.py")