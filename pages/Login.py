import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import nest_asyncio
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from google.auth.exceptions import RefreshError

# ----------------------------
# Patch asyncio for Streamlit
# ----------------------------
nest_asyncio.apply()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Stock Broke - Login", page_icon="🔐",
                   initial_sidebar_state="collapsed", layout="centered")

# ----------------------------
# Firebase Initialization
# ----------------------------
service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    initialize_app(cred)

# ----------------------------
# Gmail SMTP Setup
# ----------------------------
EMAIL_ADDRESS = st.secrets["EMAIL"]["address"]
EMAIL_PASSWORD = st.secrets["EMAIL"]["password"]

def send_verification_email(user_email):
    action_code_settings = auth.ActionCodeSettings(
        url=st.secrets["APP_URL"],  # Deployed app URL
        handle_code_in_app=True
    )
    link = auth.generate_email_verification_link(user_email, action_code_settings)
    subject = "Verify your Stock Broke Account"
    body = f"Hi,\n\nVerify your account:\n{link}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = user_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())

def send_password_reset_email(user_email):
    action_code_settings = auth.ActionCodeSettings(
        url=st.secrets["APP_URL"],  # After reset, redirect here
        handle_code_in_app=True
    )
    reset_link = auth.generate_password_reset_link(user_email, action_code_settings)
    subject = "Reset your Stock Broke Password"
    body = f"Hi,\n\nReset your password:\n{reset_link}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = user_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())
    st.success("Password reset link sent! Check your inbox.")

# ----------------------------
# Google OAuth Setup
# ----------------------------
client_id = st.secrets["google_oauth"]["client_id"]
client_secret = st.secrets["google_oauth"]["client_secret"]
redirect_url = "https://stock-broke.streamlit.app/Login" # e.g. https://share.streamlit.io/<username>/<repo>/main/Login
client = GoogleOAuth2(client_id, client_secret)

async def get_access_token(client: GoogleOAuth2, redirect_url: str, code: str):
    return await client.get_access_token(code, redirect_url)

async def get_email(client: GoogleOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email

def get_logged_in_user_email():
    try:
        params = st.experimental_get_query_params()
        code = params.get("code")
        if code:
            code = code[0]
            token = asyncio.run(get_access_token(client, redirect_url, code))
            st.experimental_set_query_params()  # Clear code
            if token:
                user_id, user_email = asyncio.run(get_email(client, token["access_token"]))
                if user_email:
                    try:
                        user = auth.get_user_by_email(user_email)
                    except exceptions.FirebaseError as e:
                        if "not found" in str(e).lower():
                            user = auth.create_user(email=user_email)
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

def show_google_login_button():
    try:
        authorization_url = asyncio.run(
            client.get_authorization_url(
                redirect_url,
                scope=["email", "profile"],
                extras_params={"access_type": "offline"},
            )
        )
        button_html = f'''
        <a href="{authorization_url}" target="_self" style="text-decoration:none;">
            <div style='
                display:flex; align-items:center; justify-content:center;
                background:white; color:black; border:1px solid #d6d6d6;
                border-radius:0.5rem; font-size:0.9rem; font-weight:500;
                padding:0.6rem 1.2rem; width:250px; margin:10px auto;
                box-shadow:0 1px 3px rgba(0,0,0,0.1); cursor:pointer;'>
                <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
                     style="height:20px;width:20px;margin-right:10px;">
                <span>Login with Google</span>
            </div>
        </a>
        '''
        st.markdown(button_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to generate Google auth URL: {e}")

# ----------------------------
# Session State Defaults
# ----------------------------
if "username" not in st.session_state: st.session_state.username = ""
if "usermail" not in st.session_state: st.session_state.usermail = ""
if "singedout" not in st.session_state: st.session_state.singedout = False
if "singout" not in st.session_state: st.session_state.singout = False

# ----------------------------
# Auth Functions
# ----------------------------
def login_callback():
    email = st.session_state.get("input_email", "").strip()
    if not email: st.warning("Enter an email."); return
    try:
        user = auth.get_user_by_email(email)
        if not user.email_verified:
            st.error("Email not verified!")
            return
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        st.success("Login successful!")
    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower(): st.warning("User not found.")
        else: st.error(f"Login error: {e}")

def signup_callback():
    email = st.session_state.get("signup_email", "").strip()
    password = st.session_state.get("signup_password", "").strip()
    username = st.session_state.get("signup_username", "").strip()
    if not all([email, password]): st.warning("Fill in email and password."); return
    try:
        user = auth.create_user(email=email, password=password, uid=username)
        send_verification_email(email)
        st.success("Account created! Check inbox to verify.")
    except Exception as e: st.error(f"Signup error: {e}")

def reset_password_callback():
    email = st.session_state.reset_email.strip()
    if not email: st.warning("Enter email."); return
    send_password_reset_email(email)

def logout_callback():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ""
    st.session_state.usermail = ""

# ----------------------------
# Check Google login at start
# ----------------------------
google_email = get_logged_in_user_email()
if google_email and not st.session_state.singedout:
    try:
        user = auth.get_user_by_email(google_email)
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
    except exceptions.FirebaseError:
        st.error("Failed to fetch Google user.")

# ----------------------------
# UI
# ----------------------------
st.title("Welcome to Stock Broke!")

if st.session_state.singout:
    st.subheader("Welcome Back!")
    st.text(f"Username: {st.session_state.username}")
    st.text(f"Email: {st.session_state.usermail}")
    st.button("SignOut", on_click=logout_callback)
else:
    choice = st.selectbox("Login/SignUp", ["Login", "SignUp", "Reset Password"])
    if choice == "Login":
        st.text_input("Email", key="input_email")
        st.text_input("Password", type="password", key="input_password")
        c1, c2, c3 = st.columns([1, 0.7, 1])
        with c1:
            st.button("Login", on_click=login_callback)
        with c3:
            show_google_login_button()
    elif choice == "SignUp":
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")
        st.text_input("Username", key="signup_username")
        st.button("SignUp", on_click=signup_callback)
    else:
        st.text_input("Enter your email", key="reset_email")
        st.button("Send Reset Link", on_click=reset_password_callback)
