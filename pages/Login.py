import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import traceback

# ==============================
# Page Config (Same as Original)
# ==============================
st.set_page_config(
    page_title="Stock Broke - Login",
    page_icon="🔐",
    initial_sidebar_state="collapsed",
    layout="centered"
)

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
            url="https://stock-broke.streamlit.app/Login",
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

def send_password_reset_email(user_email):
    try:
        action_code_settings = auth.ActionCodeSettings(
            url="https://stock-broke.streamlit.app/Login",
            handle_code_in_app=True
        )
        reset_link = auth.generate_password_reset_link(user_email, action_code_settings)
        subject = "Reset your Stock Broke Password"
        body = f"""
        Hi,

        You requested to reset your password.
        Click the link below to set a new password:

        {reset_link}

        If you did not request this, you can ignore this email.
        """
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = user_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())
        st.success("Password reset link sent! Please check your inbox.")
    except Exception as e:
        st.error(f"Failed to send reset link: {str(e)}\n{traceback.format_exc()}")

# ==============================
# Session State (Same as Original)
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
        st.session_state.singout = True
        st.session_state.singedout = True
        st.session_state.input_email = ""
        st.session_state.input_password = ""
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
        user = auth.create_user(email=email, password=password, uid=username)
        send_verification_email(email)
        st.success("Account created! Please check your inbox to verify your email.")
        st.info("Go to Login after verifying.")
        st.session_state.signup_email = ""
        st.session_state.signup_password = ""
        st.session_state.signup_username = ""
        st.rerun()
    except Exception as e:
        st.error(f"Signup error: {str(e)}\n{traceback.format_exc()}")

def reset_password_callback():
    try:
        email = st.session_state.get("reset_email", "").strip()
        if not email:
            st.warning("Please enter your email.")
            return
        send_password_reset_email(email)
        st.session_state.reset_email = ""
        st.rerun()
    except Exception as e:
        st.error(f"Reset password error: {str(e)}\n{traceback.format_exc()}")

def post_google_login():
    try:
        google_email = st.user.email
        user = auth.get_user_by_email(google_email)
        if not user.email_verified:
            send_verification_email(google_email)
            st.warning("Please verify your email via the link sent.")
        st.session_state.username = user.uid
        st.session_state.usermail = google_email
        st.session_state.singout = True
        st.session_state.singedout = True
        st.query_params["page"] = "/Login"  # Redirect to /Login
        st.rerun()
    except exceptions.FirebaseError as e:
        if "not found" in str(e).lower():
            user = auth.create_user(email=google_email)
            send_verification_email(google_email)
            st.success("Google account linked to Stock Broke! Verify your email.")
            st.session_state.username = user.uid
            st.session_state.usermail = google_email
            st.session_state.singout = True
            st.session_state.singedout = True
            st.query_params["page"] = "/Login"
            st.rerun()
        else:
            st.error(f"Firebase Google sync error: {str(e)}\n{traceback.format_exc()}")
    except Exception as e:
        st.error(f"Google post-login error: {str(e)}\n{traceback.format_exc()}")

# ==============================
# UI
# ==============================
st.title("Welcome to Stock Broke!")

# Handle Google OAuth callback
if st.query_params.get("page") == "/oauth2callback" and st.user.is_logged_in:
    post_google_login()

# Main UI
if st.session_state.singout:
    st.subheader("Welcome Back!")
    st.text(f"Username: {st.session_state.username}")
    st.text(f"Email: {st.session_state.usermail}")
    if st.button("Go to Stock Analysis", type="primary"):
        st.switch_page("pages/Stock_Analysis.py")
    st.button("SignOut", on_click=lambda: [st.logout(), st.session_state.clear(), st.query_params.clear(), st.rerun()])
else:
    try:
        choice = st.selectbox("Login/SignUp", ["Login", "SignUp", "Reset Password"])

        if choice == "Login":
            st.subheader("Login Section")
            st.text_input("Email", key="input_email")
            st.text_input("Password", type="password", key="input_password")
            c1, c2, c3 = st.columns([1, 0.7, 1])
            with c1:
                if st.button("Login", on_click=login_callback):
                    st.switch_page("pages/Stock_Analysis.py")
            with c3:
                # Custom Google button (same as original)
                google_button = f'''
                <a href="#" target="_self" style="text-decoration:none;" onclick="document.getElementById('google-login').click();">
                    <div style='
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #ffffff;
                        color: #000000;
                        border: 1px solid #d6d6d6;
                        border-radius: 0.5rem;
                        font-size: 0.9rem;
                        font-weight: 500;
                        padding: 0.6rem 1.2rem;
                        width: 250px;
                        margin: 10px auto;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        cursor: pointer;
                        transition: all 0.2s ease-in-out;
                    '
                    onmouseover="this.style.backgroundColor='#f1f1f1'; this.style.boxShadow='0 2px 6px rgba(0,0,0,0.15);'" 
                    onmouseout="this.style.backgroundColor='#ffffff'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';">
                        <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" 
                             style="height:20px;width:20px;margin-right:10px;" />
                        <span>Login with Google</span>
                    </div>
                </a>
                <button id="google-login" style="display:none;" onclick="{st.login()}"></button>
                '''
                st.markdown(google_button, unsafe_allow_html=True)

        elif choice == "SignUp":
            st.subheader("Create New Account")
            st.text_input("Email", key="signup_email")
            st.text_input("Password", type="password", key="signup_password")
            st.text_input("Username", key="signup_username")
            if st.button("SignUp", on_click=signup_callback):
                pass

        elif choice == "Reset Password":
            st.subheader("Reset Password")
            st.text_input("Enter your email to reset password", key="reset_email")
            if st.button("Send Reset Link", on_click=reset_password_callback):
                pass

    except Exception as e:
        st.error(f"App error: {str(e)}\n{traceback.format_exc()}")
