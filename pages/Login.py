import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import traceback

# ==============================
# Firebase Initialization
# ==============================
@st.cache_resource
def init_firebase():
    try:
        service_account_str = st.secrets["FIREBASE"]["json"]
        service_account_info = json.loads(service_account_str)
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_info)
            initialize_app(cred)
        return True
    except Exception as e:
        st.error(f"Firebase init failed: {str(e)}\n{traceback.format_exc()}")
        return False

if not init_firebase():
    st.stop()

# ==============================
# Gmail SMTP Setup
# ==============================
EMAIL_ADDRESS = st.secrets["EMAIL"]["address"]
EMAIL_PASSWORD = st.secrets["EMAIL"]["password"]

def send_verification_email(user_email):
    try:
        action_code_settings = auth.ActionCodeSettings(
            url="https://stock-broke.streamlit.app/Login",  # Redirect to /Login
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
    except Exception as e:
        st.error(f"Email sending failed: {str(e)}\n{traceback.format_exc()}")

# ==============================
# Session State Defaults
# ==============================
if "username" not in st.session_state:
    st.session_state.username = ""
if "usermail" not in st.session_state:
    st.session_state.usermail = ""
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "redirect_to_login" not in st.session_state:
    st.session_state.redirect_to_login = False

# ==============================
# Auth Functions
# ==============================
def login_callback():
    try:
        email = st.session_state.get("input_email", "").strip()
        if not email:
            st.warning("Please enter an email.")
            return
        user = auth.get_user_by_email(email)
        if not user.email_verified:
            st.error("Email not verified! Please check your inbox.")
            return
        st.success("Login successful!")
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.is_logged_in = True
        st.session_state.input_email = ""
        st.session_state.input_password = ""
        st.session_state.redirect_to_login = False
        st.rerun()
    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower():
            st.warning("User not found. Please sign up first.")
        else:
            st.error(f"Login error: {str(e)}\n{traceback.format_exc()}")
    except Exception as e:
        st.error(f"Unexpected login error: {str(e)}\n{traceback.format_exc()}")

def signup_callback():
    try:
        email = st.session_state.get("signup_email", "").strip()
        password = st.session_state.get("signup_password", "").strip()
        username = st.session_state.get("signup_username", "").strip()
        if not all([email, password]):
            st.warning("Fill in email and password.")
            return
        user = auth.create_user(email=email, password=password)
        send_verification_email(email)
        st.success("Account created! Please check your inbox to verify your email.")
        st.info("Go to Login after verifying.")
        st.session_state.signup_email = ""
        st.session_state.signup_password = ""
        st.session_state.signup_username = ""
        st.session_state.redirect_to_login = True
        st.rerun()
    except Exception as e:
        st.error(f"Signup error: {str(e)}\n{traceback.format_exc()}")

def post_google_login():
    try:
        google_email = st.user.email
        user = auth.get_user_by_email(google_email)
        if not user.email_verified:
            send_verification_email(google_email)
            st.warning("Please verify your email via the link sent.")
        st.session_state.username = user.uid
        st.session_state.usermail = google_email
        st.session_state.is_logged_in = True
        st.session_state.redirect_to_login = True  # Flag to redirect to /Login
        st.rerun()
    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower():
            user = auth.create_user(email=google_email)
            send_verification_email(google_email)
            st.success("Google account linked to Stock Broke! Verify your email.")
            st.session_state.username = user.uid
            st.session_state.usermail = google_email
            st.session_state.is_logged_in = True
            st.session_state.redirect_to_login = True
            st.rerun()
        else:
            st.error(f"Firebase Google sync error: {str(e)}\n{traceback.format_exc()}")
    except Exception as e:
        st.error(f"Google post-login error: {str(e)}\n{traceback.format_exc()}")

# ==============================
# UI and Routing
# ==============================
st.title("Welcome to Stock Broke!")

# Detect current page (based on URL path)
current_path = st.query_params.get("page", "/")
if current_path == "/oauth2callback" and st.user.is_logged_in:
    # After Google OAuth, redirect to /Login
    post_google_login()
elif current_path == "/Login" or st.session_state.redirect_to_login:
    try:
        if not st.user.is_logged_in and not st.session_state.is_logged_in:
            choice = st.selectbox("Login/SignUp", ["Login", "SignUp", "Google Login"])

            if choice == "Login":
                st.subheader("Login Section")
                st.text_input("Email", key="input_email")
                st.text_input("Password", type="password", key="input_password")
                st.button("Login", on_click=login_callback)

            elif choice == "SignUp":
                st.subheader("Create New Account")
                st.text_input("Email", key="signup_email")
                st.text_input("Password", type="password", key="signup_password")
                st.text_input("Username", key="signup_username")
                st.button("SignUp", on_click=signup_callback)

            else:  # Google Login
                st.subheader("Or use Google:")
                st.header("This app is private.")
                st.subheader("Please log in.")
                st.button("Log in with Google", on_click=st.login)

        else:
            # Logged-in user UI
            st.subheader("Welcome Back!")
            st.text(f"Username: {st.session_state.username}")
            st.text(f"Email: {st.session_state.usermail}")
            if st.button("Go to Stock Analysis", type="primary"):
                st.switch_page("pages/Stock_Analysis.py")
            st.button("Log out", on_click=lambda: [st.logout(), st.session_state.clear(), st.rerun()])
    except Exception as e:
        st.error(f"App error: {str(e)}\n{traceback.format_exc()}")
else:
    # Default route: Redirect to /Login
    st.query_params["page"] = "/Login"
    st.rerun()
