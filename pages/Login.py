import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from google.auth.exceptions import RefreshError



# ==============================
# Firebase Initialization
# ==============================
service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    initialize_app(cred)

# ==============================
# Gmail SMTP Setup
# ==============================
EMAIL_ADDRESS = st.secrets["EMAIL"]["address"]
EMAIL_PASSWORD = st.secrets["EMAIL"]["password"]

def send_verification_email(user_email):
    action_code_settings = auth.ActionCodeSettings(
        url="https://stock-broke.streamlit.app/",  # Update to deployed URL
        handle_code_in_app=True
    )
    try:
        link = auth.generate_email_verification_link(user_email, action_code_settings)
        subject = "Verify your Stock Broke Account"
        body = f"""
        Hi,

        Thanks for signing up for Stock Broke!
        Please verify your email by clicking this link:

        {link}

        If you don’t see this email in your inbox, please check your spam or junk folder.
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
        st.error(f"Failed to send verification email: {e}")

# ==============================
# Google OAuth Setup
# ==============================
try:
    client_id = st.secrets["google_oauth"]["client_id"]
    client_secret = st.secrets["google_oauth"]["client_secret"]
except KeyError as e:
    st.error(f"Missing Google OAuth secret: {e}. Add to secrets.toml.")
    st.stop()

redirect_url = "https://stock-broke.streamlit.app/"
client = GoogleOAuth2(client_id=client_id, client_secret=client_secret)

async def get_access_token(client: GoogleOAuth2, redirect_url: str, code: str):
    return await client.get_access_token(code, redirect_url)

async def get_email(client: GoogleOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email

def get_logged_in_user_email():
    try:
        query_params = st.query_params
        code = query_params.get("code")
        if code:
            token = asyncio.run(get_access_token(client, redirect_url, code))
            st.experimental_set_query_params()  # Clear code from URL
            if token:
                user_id, user_email = asyncio.run(get_email(client, token["access_token"]))
                if user_email:
                    try:
                        user = auth.get_user_by_email(user_email)
                    except exceptions.FirebaseError as e:
                        if "not found" in str(e).lower():
                            user = auth.create_user(email=user_email, email_verified=True)  # Mark Google users as verified
                        else:
                            raise
                    st.session_state.username = user.uid
                    st.session_state.usermail = user.email
                    st.session_state.singout = True
                    st.session_state.singedout = True
                    return user.email
        return None
    except Exception as e:
        st.error(f"Google auth error: {e}")
        return None
        
def show_login_button():
    try:
        authorization_url = asyncio.run(
            client.get_authorization_url(
                redirect_url,
                scope=["email", "profile"],
                extras_params={"access_type": "offline"},
            )
        )
        st.markdown(f'<a href="{authorization_url}" target="_self">Login with Google</a>', unsafe_allow_html=True)
        get_logged_in_user_email()
    except Exception as e:
        st.error(f"Failed to generate auth URL: {e}")

# ==============================
# Session State Defaults
# ==============================
if "username" not in st.session_state:
    st.session_state.username = ""
if "usermail" not in st.session_state:
    st.session_state.usermail = ""
if "singedout" not in st.session_state:
    st.session_state.singedout = False
if "singout" not in st.session_state:
    st.session_state.singout = False

# ==============================
# Auth Functions
# ==============================

def login_callback():
    email = st.session_state.get("input_email", "").strip()
    if not email:
        st.warning("Please enter an email.")
        return
    try:
        user = auth.get_user_by_email(email)
        st.write(f"Debug: Email verified status: {user.email_verified}")  # Debug
        is_google_user = any(provider.provider_id == "google.com" for provider in user.provider_data)
        if not user.email_verified and not is_google_user:
            st.error("Email not verified! Please check your inbox.")
            return
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        st.success("Login successful!")
        st.switch_page('Stock_Analysis')
    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower():
            st.warning("User not found. Please sign up first.")
        else:
            st.error(f"Login error: {e}")

def signup_callback():
    email = st.session_state.get("signup_email", "").strip()
    password = st.session_state.get("signup_password", "").strip()
    username = st.session_state.get("signup_username", "").strip()
    if not all([email, password]):
        st.warning("Fill in email and password.")
        return
    try:
        user = auth.create_user(email=email, password=password, uid=username)
        send_verification_email(email)
        st.success("Account created! Please check your inbox to verify your email.")
        st.info("Go to Login after verifying.")
        # Clear fields
        st.session_state.signup_email = ""
        st.session_state.signup_password = ""
        st.session_state.signup_username = ""
    except RefreshError as e:
        st.error(f"Auth failed (invalid service account): {e}. Generate a new key.")
    except Exception as e:
        st.error(f"Signup error: {e}")

def logout_callback():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ""
    st.session_state.usermail = ""

# ==============================
# UI
# ==============================
st.title("Welcome to Stock Broke!")

if not st.session_state.singedout:
    choice = st.selectbox("Login/SignUp", ["Login", "SignUp"])

    if choice == "Login":
        st.subheader("Login Section")
        st.text_input("Email", key="input_email")
        st.text_input("Password", type="password", key="input_password")
        if st.button("Login", on_click=login_callback):
            pass
        st.markdown("Or use Google:")
        show_login_button()

    else:
        st.subheader("Create New Account")
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")
        st.text_input("Username", key="signup_username")
        if st.button("SignUp", on_click=signup_callback):
            pass
        st.markdown("Or use Google:")
        show_login_button()

if st.session_state.singout:
    st.subheader("Welcome Back!")
    st.text(f"Username: {st.session_state.username}")
    st.text(f"Email: {st.session_state.usermail}")
    if st.button("SignOut", on_click=logout_callback):
        pass




