import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from google.auth.exceptions import RefreshError


st.set_page_config(page_title="Stock Broke - Login", page_icon="🔐",initial_sidebar_state="collapsed",layout="centered")


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
        url="https://stock-broke.streamlit.app/Login",  # Change to your deployed URL
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

def send_password_reset_email(user_email):
    try:
        action_code_settings = auth.ActionCodeSettings(
            url="https://stock-broke.streamlit.app/Login",  # After reset, redirect here
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
        st.error(f"Failed to send reset link: {e}")

def show_login_button():
    try:
        # Generate authorization URL
        authorization_url = asyncio.run(
            client.get_authorization_url(
                redirect_url,
                scope=["email", "profile"],
                extras_params={"access_type": "offline"},
            )
        )

        # Google-style button, matches Streamlit's native button theme
        google_button = f'''
        <a href="{authorization_url}" target="_self" style="text-decoration:none;">
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
            onmouseover="this.style.backgroundColor='#f1f1f1'; this.style.boxShadow='0 2px 6px rgba(0,0,0,0.15)';" 
            onmouseout="this.style.backgroundColor='#ffffff'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';">
                <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" 
                     style="height:20px;width:20px;margin-right:10px;" />
                <span>Login with Google</span>
            </div>
        </a>
        '''

        st.markdown(google_button, unsafe_allow_html=True)

        # Handle login state after redirect
        get_logged_in_user_email()

    except Exception as e:
        st.error(f"Failed to generate auth URL: {e}")


# ==============================
# Google OAuth Setup
# ==============================
try:    
    client_id = st.secrets["google_oauth"]["client_id"]
    client_secret = st.secrets["google_oauth"]["client_secret"]
except KeyError as e:
    st.error(f"Missing Google OAuth secret: {e}. Add to secrets.toml.")
    st.stop()

redirect_url = "https://stock-broke-f5d95.firebaseapp.com/__/auth/handler"
client = GoogleOAuth2(client_id=client_id, client_secret=client_secret)

async def get_access_token(client: GoogleOAuth2, redirect_url: str, code: str):
    return await client.get_access_token(code, redirect_url)

async def get_email(client: GoogleOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email
def get_logged_in_user_email():
    try:
        code = st.query_params.get("code")
        if code:
            token = asyncio.run(get_access_token(client, redirect_url, code))
            # Clear code from URL using new API
            st.query_params.clear()
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

        if not user.email_verified:
            st.error("Email not verified! Please check your inbox.")
            return

        st.success("Login successful!")
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        

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
        
def reset_password_callback():
    email = st.session_state.reset_email.strip()
    if not email:
        st.warning("Please enter your email.")
        return
    send_password_reset_email(email)

    st.session_state.reset_email = ""


def logout_callback():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ""
    st.session_state.usermail = ""

# ==============================
# UI
# ==============================

# At the very top, after session_state defaults
google_email = get_logged_in_user_email()
if google_email and not st.session_state.singedout:
    # Mark user as logged in
    try:
        user = auth.get_user_by_email(google_email)
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
    except exceptions.FirebaseError:
        st.error("Failed to fetch Google user from Firebase.")

st.title("Welcome to Stock Broke!")

if st.session_state.singout:
        st.subheader("Welcome Back!")
        st.text(f"Username: {st.session_state.username}")
        st.text(f"Email: {st.session_state.usermail}")
        if st.button("SignOut", on_click=logout_callback):
            pass

else:
    choice = st.selectbox("Login/SignUp", ["Login", "SignUp","Reset Password"])

    if choice == "Login":
        st.subheader("Login Section")
        st.text_input("Email", key="input_email")
        st.text_input("Password", type="password", key="input_password")
        if st.button("Login", on_click=login_callback):
            st.switch_page("pages/Stock_Analysis.py")
            
        st.markdown("Or use Google:")
        show_login_button()

    elif choice == "Signup":
        st.subheader("Create New Account")
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")
        st.text_input("Username", key="signup_username")
        if st.button("SignUp", on_click=signup_callback):
            pass
       
    else:
        st.subheader("Reset Password")
        st.text_input("Enter your email to reset password", key="reset_email")
        if st.button("Send Reset Link", on_click=reset_password_callback):
            pass







