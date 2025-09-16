import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app

# ==============================
# Firebase Initialization (Cached for Efficiency)
# ==============================
@st.cache_resource
def init_firebase():
    service_account_str = st.secrets["FIREBASE"]["json"]
    service_account_info = json.loads(service_account_str)
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        initialize_app(cred)
    return True

init_firebase()  # Run once

# ==============================
# Gmail SMTP Setup (Unchanged)
# ==============================
EMAIL_ADDRESS = st.secrets["EMAIL"]["address"]
EMAIL_PASSWORD = st.secrets["EMAIL"]["password"]

def send_verification_email(user_email):
    action_code_settings = auth.ActionCodeSettings(
        url="https://stock-broke.streamlit.app/",  # Update to your app's base URL (no /Login needed)
        handle_code_in_app=True
    )
    link = auth.generate_email_verification_link(user_email, action_code_settings)

    subject = "Verify your Stock Broke Account"
    body = f"""
    Hi,

    Thanks for signing up for Stock Broke!
    Please verify your email by clicking this link:

    {link}

    If you did not request this, ignore this email.
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = user_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())

# ==============================
# Session State Defaults (Simplified)
# ==============================
if "username" not in st.session_state:
    st.session_state.username = ""
if "usermail" not in st.session_state:
    st.session_state.usermail = ""

# ==============================
# Auth Functions (Updated for Built-in + Firebase)
# ==============================

def login_callback():
    email = st.session_state.get("input_email", "").strip()
    if not email:
        st.warning("Please enter an email.")
        return
    try:
        user = auth.get_user_by_email(email)
        if not user.email_verified:
            st.error("Email not verified! Please check your inbox.")
            return

        st.success("Login successful!")
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.rerun()  # Refresh to show logged-in UI

        # Clear inputs
        st.session_state.input_email = ""
        st.session_state.input_password = ""

    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower():
            st.warning("User not found. Please sign up first.")
        else:
            st.error(f"Login error: {e}")
    except Exception as e:
        st.error(f"Unexpected login error: {e}")

def signup_callback():
    email = st.session_state.get("signup_email", "").strip()
    password = st.session_state.get("signup_password", "").strip()
    username = st.session_state.get("signup_username", "").strip()
    if not all([email, password]):
        st.warning("Fill in email and password.")
        return
    try:
        user = auth.create_user(email=email, password=password)  # Note: uid auto-generated; use username if needed via display_name
        send_verification_email(email)
        st.success("Account created! Please check your inbox to verify your email.")
        st.info("Go to Login after verifying.")
        # Clear fields
        st.session_state.signup_email = ""
        st.session_state.signup_password = ""
        st.session_state.signup_username = ""
    except Exception as e:
        st.error(f"Signup error: {e}")

def post_google_login():
    """Handle Firebase user after Google login (create if missing, verify email if needed)"""
    google_email = st.user.email
    try:
        user = auth.get_user_by_email(google_email)
        # If not verified, prompt to verify (but Google emails are usually verified)
        if not user.email_verified:
            st.warning("Please verify your email if prompted.")
    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower():
            # Create user from Google auth
            user = auth.create_user(email=google_email)
            st.success("Google account linked to Stock Broke!")
        else:
            raise
    st.session_state.username = user.uid
    st.session_state.usermail = user.email
    st.rerun()

# ==============================
# UI (Using Built-in st.login/st.user)
# ==============================
st.title("Welcome to Stock Broke!")

# Built-in Google Auth Check
if not st.user.is_logged_in:
    # Show manual login/signup or Google button
    choice = st.selectbox("Login/SignUp", ["Login", "SignUp", "Google Login"])

    if choice == "Login":
        st.subheader("Login Section")
        st.text_input("Email", key="input_email")
        st.text_input("Password", type="password", key="input_password")
        if st.button("Login", on_click=login_callback):
            st.switch_page("pages/Stock_Analysis.py")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")
        st.text_input("Username", key="signup_username")
        if st.button("SignUp", on_click=signup_callback):
            pass

    else:  # Google Login
        st.subheader("Or use Google:")
        def login_screen():
            st.header("This app is private.")
            st.subheader("Please log in.")
            st.button("Log in with Google", on_click=st.login)
        login_screen()

else:
    # Post-Google login: Sync with Firebase
    if st.session_state.usermail != st.user.email:  # First-time Google login
        post_google_login()

    st.subheader("Welcome Back!")
    st.text(f"Username: {st.session_state.username}")
    st.text(f"Email: {st.user.email}")  # Use st.user for Google details
    if st.button("Go to Stock Analysis", type="primary"):
        st.switch_page("pages/Stock_Analysis.py")
    st.button("Log out", on_click=st.logout)
