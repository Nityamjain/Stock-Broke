
import streamlit as st
import requests
import json
import smtplib
from email.mime.text import MIMEText
from firebase_admin import auth, exceptions
import nest_asyncio
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from google.auth.exceptions import RefreshError
import firebase_admin
from firebase_admin import credentials, auth,firestore

# -------------------------------------------------------------------
# Initialization & secrets validation
# -------------------------------------------------------------------
import json 

service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Validate secrets
try:
    EMAIL_ADDRESS = st.secrets["EMAIL"]["address"]
    EMAIL_PASSWORD = st.secrets["EMAIL"]["password"]
except Exception as e:
    st.error("Missing EMAIL secrets in secrets.toml. Add EMAIL.address and EMAIL.password.")
    st.stop()

try:
    GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
except Exception as e:
    st.error("Missing google_oauth secrets in secrets.toml. Add google_oauth.client_id and client_secret.")
    st.stop()

try:

    FIREBASE_API_KEY = st.secrets ["FIREBASE_WEB"]["apiKey"]
except Exception:
    FIREBASE_API_KEY = None
    st.warning("FIREBASE.api_key not found in secrets.toml. Email/password sign-in will be disabled.")

# Google OAuth client
redirect_url = "http://localhost:8501/Login"  # Change for production
client = GoogleOAuth2(client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET)

# Apply nest_asyncio to avoid 'asyncio.run() cannot be called from a running event loop' inside Streamlit
try:
    nest_asyncio.apply()
except Exception:
    pass

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def get_first_query_param(key, default=None):
    """Return the first entry of st.query_params[key] (Streamlit gives list values)."""
    qp = st.query_params.get(key)
    if not qp:
        return default
    if isinstance(qp, (list, tuple)):
        return qp[0]
    return qp

def clear_query_params():
    """Clear query params in browser URL."""
    st.query_params.clear()

def run_async(coro):
    """
    Safely run coroutine inside Streamlit.
    If event loop already running, schedule coroutine and wait for result.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    else:
        return loop.run_until_complete(coro)

def smtp_send(from_addr, to_addr, subject, body):
    """Send a simple plaintext email via Gmail SMTP (SSL)."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_addr, EMAIL_PASSWORD)
        server.sendmail(from_addr, to_addr, msg.as_string())

# -------------------------------------------------------------------
# Email verification & password reset (Admin SDK links + SMTP)
# -------------------------------------------------------------------
def send_verification_email(user_email):
    try:
        action_code_settings = auth.ActionCodeSettings(
            url="http://localhost:8051/Login",
            handle_code_in_app=True
        )
        link = auth.generate_email_verification_link(user_email, action_code_settings)
        subject = "Verify your Stock Broke Account"
        body = f"""Hi,

Thanks for signing up for Stock Broke!
Please verify your email by clicking this link:

{link}

If you did not request this, ignore this email.
"""
        smtp_send(EMAIL_ADDRESS, user_email, subject, body)
    except Exception as e:
        st.error(f"Failed to send verification email: {str(e)}")

def send_password_reset_email(user_email):
    try:
        action_code_settings = auth.ActionCodeSettings(
            url="http://localhost:8051/Login",
            handle_code_in_app=True
        )
        link = auth.generate_password_reset_link(user_email, action_code_settings)
        subject = "Reset your Stock Broke Password"
        body = f"""Hi,

You requested a password reset for your Stock Broke account.
Click the link below to reset your password:

{link}

If you did not request this, please ignore this email.
"""
        smtp_send(EMAIL_ADDRESS, user_email, subject, body)
    except Exception as e:
        st.error(f"Failed to send password reset email: {str(e)}")

# -------------------------------------------------------------------
# Firebase REST password sign-in
# -------------------------------------------------------------------
def firebase_sign_in_with_email_and_password(email: str, password: str):
    """
    Sign in using Firebase Identity Toolkit REST API to validate email+password.
    Returns dict with idToken, localId, email on success, else None.
    Requires FIREBASE_API_KEY in secrets.
    """
    if not FIREBASE_API_KEY:
        st.error("FIREBASE.api_key not set. Cannot sign in with password.")
        return None

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        res = requests.post(url, json=payload, timeout=10)
        data = res.json()
        if res.status_code == 200:
            return data
        else:
            # Example error: {"error":{"message":"EMAIL_NOT_FOUND"}}
            return {"error": data.get("error", {}).get("message", str(data))}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# Google OAuth helpers
# -------------------------------------------------------------------
async def _get_access_token_async(code: str):
    return await client.get_access_token(code, redirect_url)

async def _get_id_email_async(access_token: str):
    return await client.get_id_email(access_token)

def get_access_token_sync(code: str):
    try:
        return run_async(_get_access_token_async(code))
    except Exception as e:
        st.error(f"Failed to retrieve Google OAuth token: {str(e)}")
        return None

def get_id_email_sync(access_token: str):
    try:
        return run_async(_get_id_email_async(access_token))
    except Exception as e:
        st.error(f"Failed to retrieve email from Google: {str(e)}")
        return (None, None)

def generate_google_authorization_url(state: str):
    """Generate Google OAuth authorization URL safely (sync)."""
    try:
        # get_authorization_url may be coroutine or sync depending on version; handle both cases
        try:
            url = run_async(client.get_authorization_url(
                redirect_url,
                scope=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
                extras_params={"access_type": "offline", "state": state},
            ))
        except TypeError:
            url = client.get_authorization_url(
                redirect_url,
                scope=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
                extras_params={"access_type": "offline", "state": state},
            )
        return url
    except Exception as e:
        st.error(f"Failed to generate auth URL: {e}")
        return None

# -------------------------------------------------------------------
# Streamlit session-state defaults
# -------------------------------------------------------------------
if "username" not in st.session_state:
    st.session_state.username = ""
if "usermail" not in st.session_state:
    st.session_state.usermail = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "view" not in st.session_state:
    st.session_state.view = "Login"
if "oauth_processing" not in st.session_state:
    st.session_state.oauth_processing = False

# -------------------------------------------------------------------
# Core flows for Google sign-in/sign-up processing
# -------------------------------------------------------------------
def process_google_flow(expected_state="login"):
    """
    Processes Google OAuth redirect for either login or signup.
    Returns (user_id, user_email) on success, else None.
    """
    if st.session_state.get("oauth_processing", False):
        return None
    st.session_state.oauth_processing = True
    try:
        code = get_first_query_param("code")
        state = get_first_query_param("state")
        if code and state == expected_state:
            token = get_access_token_sync(code)
            if not token:
                return None
            # token may be dict with access_token
            access_token = token.get("access_token") if isinstance(token, dict) else token
            user_id, user_email = get_id_email_sync(access_token)
            return (user_id, user_email)
        return None
    finally:
        st.session_state.oauth_processing = False

# -------------------------------------------------------------------
# UI actions: login, signup, logout handlers
# -------------------------------------------------------------------
def login_with_email_password_callback():
    email = st.session_state.get("input_email", "").strip()
    password = st.session_state.get("input_password", "").strip()
    if not email or not password:
        st.warning("Please enter both email and password.")
        return

    signin = firebase_sign_in_with_email_and_password(email, password)
    if signin is None:
        return
    if signin.get("error"):
        err = signin.get("error")
        # Map some common Firebase errors to friendly messages:
        if "EMAIL_NOT_FOUND" in str(err):
            st.error("No account found with that email. Please sign up first.")
            return
        elif "INVALID_PASSWORD" in str(err):
            st.error("Invalid password. Try again.")
            return
        else:
            st.error(f"Sign-in failed: {err}")
            return

    # At this point, credential validated by Firebase
    try:
        user = auth.get_user_by_email(email)
        if not user.email_verified:
            st.error("Email not verified! Please check your inbox (or resend verification).")
            return
        st.success("Login successful!")
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.logged_in = True
        st.session_state.view = "LoggedIn"
        st.session_state.input_email = ""
        st.session_state.input_password = ""
        clear_query_params()
        st.rerun()

    except exceptions.FirebaseError as e:
        st.error(f"Firebase admin error: {e}")
    except Exception as e:
        st.error(f"Unexpected login error: {e}")

def signup_with_email_password_callback():
    email = st.session_state.get("signup_email", "").strip()
    password = st.session_state.get("signup_password", "").strip()
    username = st.session_state.get("signup_username", "").strip()
    if not email or not password:
        st.warning("Fill in email and password.")
        return
    try:
        if username:
            user = auth.create_user(email=email, password=password, uid=username)
        else:
            user = auth.create_user(email=email, password=password)  # Firebase generates uid
        send_verification_email(email)
        st.success("Account created! Verification email sent. Please verify and then login.")
        st.session_state.signup_email = ""
        st.session_state.signup_password = ""
        st.session_state.signup_username = ""
    except RefreshError as e:
        st.error(f"Auth failed (invalid service account): {e}. Generate a new key for the admin SDK.")
    except Exception as e:
        st.error(f"Signup error: {e}")

def logout_callback():
    st.session_state.logged_in = False
    st.session_state.view = "Login"
    st.session_state.username = ""
    st.session_state.usermail = ""
    clear_query_params()
    
    


# -------------------------------------------------------------------
# Simple UI helper to show Google buttons
# -------------------------------------------------------------------
def show_google_button(is_signup=False):
    state = "signup" if is_signup else "login"
    url = generate_google_authorization_url(state=state)
    if not url:
        st.error("Could not generate Google authorization URL.")
        return
    btn_text = "Sign Up with Google" if is_signup else "Login with Google"
    st.markdown(f'<a href="{url}" target="_self" style="text-decoration:none"><button>{btn_text}</button></a>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Page rendering: main UI
# -------------------------------------------------------------------
st.title("Welcome to Stock Broke!")

# If not logged in, check for Google OAuth redirect
if not st.session_state.logged_in:
    if st.session_state.oauth_processing:
        with st.spinner("Processing Google authentication..."):
            pass
    else:
        # If view is Login or SignUp, process any redirect
        if st.session_state.view == "Login":
            result = process_google_flow(expected_state="login")
            if result:
                user_id, user_email = result
                try:
                    user = auth.get_user_by_email(user_email)
                    if not user.email_verified:
                        st.error("Email not verified! Please check your inbox.")
                    else:
                        st.session_state.username = user.uid
                        st.session_state.usermail = user.email
                        st.session_state.logged_in = True
                        st.session_state.view = "LoggedIn"
                        clear_query_params()
                        st.rerun()

                except exceptions.FirebaseError as e:
                    if "user-not-found" in str(e).lower():
                        st.warning("User not found. Please sign up first.")
                    else:
                        st.error(f"Firebase lookup error: {e}")

        elif st.session_state.view == "SignUp":
            result = process_google_flow(expected_state="signup")
            if result:
                user_id, user_email = result
                try:
                    # If user exists, ask them to login instead
                    existing = auth.get_user_by_email(user_email)
                    st.warning("Account already exists. Please log in instead.")
                    st.session_state.view = "Login"
                    clear_query_params()
                except exceptions.FirebaseError as e:
                    if "user-not-found" in str(e).lower():
                        # Create user record with Google UID (if UID valid)
                        try:
                            new_user = auth.create_user(email=user_email, email_verified=True, uid=user_id)
                            st.session_state.username = new_user.uid
                            st.session_state.usermail = new_user.email
                            st.session_state.logged_in = True
                            st.session_state.view = "LoggedIn"
                            st.success(f"Account created and logged in successfully for {user_email}!")
                            clear_query_params()
                            st.rerun()

                        except Exception as create_error:
                            st.error(f"Failed to create user: {create_error}")
                    else:
                        st.error(f"Firebase error checking user: {e}")

# Main auth UI
if not st.session_state.logged_in:
    # Auth mode selector
    st.selectbox(
        "Login/SignUp",
        ["Login", "SignUp", "Reset Password"],
        key="auth_view_select",
        on_change=lambda: st.session_state.__setitem__("view", st.session_state.auth_view_select),
        disabled=st.session_state.oauth_processing
    )

    if st.session_state.view == "Login":
        st.subheader("Login")
        st.text_input("Email", key="input_email")
        st.text_input("Password", type="password", key="input_password")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Login", on_click=login_with_email_password_callback)
        with col2:
            st.button("Forgot Password?", on_click=lambda: st.session_state.__setitem__("view", "Reset Password"))
        st.divider()
        st.markdown("<p style='text-align:center;'>Or</p>", unsafe_allow_html=True)
        show_google_button(is_signup=False)

    elif st.session_state.view == "SignUp":
        st.subheader("Create New Account")
        st.text_input("Username (optional)", key="signup_username")
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")
        st.button("SignUp", on_click=signup_with_email_password_callback)


    elif st.session_state.view == "Reset Password":
        st.subheader("Reset Password")
        st.text_input("Enter your registered email", key="reset_email")
        if st.button("Send Reset Email"):
            email = st.session_state.get("reset_email", "").strip()
            if email:
                try:
                    auth.get_user_by_email(email)  # verify user exists
                    send_password_reset_email(email)
                    st.success("Password reset email sent! Check your inbox.")
                    st.session_state.reset_email = ""
                except exceptions.FirebaseError as e:
                    if "user-not-found" in str(e).lower():
                        st.error("No account found with that email address.")
                    else:
                        st.error(f"An error occurred: {e}")
                except Exception as e:
                    st.error(f"Failed to send reset email: {e}")
            else:
                st.warning("Please enter your email.")
        st.button("Back to Login", on_click=lambda: st.session_state.__setitem__("view", "Login"))

else:
    # Post-login UI
    st.subheader(f"Welcome, {st.session_state.username}!")
    st.text(f"Email: {st.session_state.usermail}")
    st.button("SignOut", on_click=logout_callback)
    # Navigation links (adjust based on your app structure)
    try:
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/Stock_Analysis.py", label="Stock Analysis", icon="📈")
        st.page_link("pages/CAPM.py", label="CAPM & Returns", icon="💼")
        st.page_link("pages/Stock_Prediction.py", label="Prediction", icon="🔍")
        st.page_link("pages/Watchlist.py", label="Watchlist", icon="⚙️")
    except Exception:
        # streamlit's page_link has particular structure; ignore if missing in older versions
        st.write("Navigate using the sidebar or app menu.")


if st.session_state.get("trigger_rerun", False):
    st.session_state.trigger_rerun = False
    st.rerun()
# -------------------------------------------------------------------
# End of module
# -------------------------------------------------------------------


