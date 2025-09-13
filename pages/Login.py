import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions

import streamlit as st
from firebase_admin import auth

# --- Session State Setup ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "usermail" not in st.session_state:
    st.session_state["usermail"] = ""

# ==============================
# Firebase Initialization
# ==============================
service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

# ==============================
# Email Setup (Gmail SMTP)
# ==============================
EMAIL_ADDRESS = st.secrets["EMAIL"]["address"]
EMAIL_PASSWORD = st.secrets["EMAIL"]["password"]

def send_verification_email(user_email):
    # Generate Firebase email verification link
    action_code_settings = auth.ActionCodeSettings(
        url="https://stock-broke.streamlit.app/Login",  # change to your deployed app URL
        handle_code_in_app=True
    )
    link = auth.generate_email_verification_link(user_email, action_code_settings)

    # Email content
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

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())

# ==============================
# Signup Function
# ==============================
def signup_callback():
    email = st.session_state.get("signup_email", "").strip()
    password = st.session_state.get("signup_password", "").strip()

    if not email or not password:
        st.warning("Enter both email and password.")
        return

    try:
        user = auth.create_user(email=email, password=password)
        send_verification_email(email)  # Send email
        st.success("Account created! Please check your inbox to verify your email.")
    except Exception as e:
        st.error(f"Signup error: {e}")

# ==============================
# Login Function
# ==============================
def login_callback():
    email = st.session_state.get("input_email", "").strip()
    password = st.session_state.get("input_password", "").strip()  # NOTE: Firebase Admin SDK does not verify password!

    if not email:
        st.warning("Enter your email.")
        return

    try:
        user = auth.get_user_by_email(email)

        if not user.email_verified:
            st.error("Email not verified! Please check your inbox.")
            return

        st.success("Login successful!")
        st.session_state.logged_in = True
        st.session_state.usermail = user.email
        st.session_state.username = user.uid

    except exceptions.FirebaseError as e:
        st.error(f"Login error: {e}")

# ==============================
# UI
# ==============================
st.title("Stock Broke - Auth System")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    choice = st.selectbox("Choose:", ["Login", "SignUp"])

    if choice == "Login":
        st.subheader("Login")
        email=st.text_input("Email", key="input_email")
        password=st.text_input("Password", type="password", key="input_password")
       
        if st.button("Login"):
            try:
                # Check if user exists in Firebase
                user = auth.get_user_by_email(email)
                # Save session
                st.session_state["authenticated"] = True
                st.session_state["username"] = user.uid
                st.session_state["usermail"] = email
        
                st.success("✅ Login successful!")
                st.switch_page("pages/Home.py")   # redirect to Home
            except Exception as e:
                st.error(f" Login failed: {e}")


    else:
        st.subheader("Sign Up")
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")
        if st.button("SignUp", on_click=signup_callback):
            pass
else:
    st.subheader("Welcome Back!")
    st.write(f"Email: {st.session_state.usermail}")
    if st.button("Logout"):
        st.session_state.logged_in = False






