import streamlit as st
import json
from google.auth.exceptions import RefreshError
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2

# Firebase credentials
service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    initialize_app(cred)

# Google OAuth
try:
    client_id = st.secrets["google_oauth"]["client_id"]
    client_secret = st.secrets["google_oauth"]["client_secret"]
except KeyError as e:
    st.error(f"Missing Google OAuth secret: {e}. Add to secrets.toml.")
    st.stop()

redirect_url = "http://localhost:8501/"  # Update to your production URL (e.g., https://your-app.streamlit.app/)
client = GoogleOAuth2(client_id=client_id, client_secret=client_secret)

async def get_access_token(client: GoogleOAuth2, redirect_url: str, code: str):
    return await client.get_access_token(code, redirect_url)

async def get_email(client: GoogleOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email

def get_logged_in_user_email():
    try:
        query_params = st.query_params
        code = query_params.get('code')
        if code:
            token = asyncio.run(get_access_token(client, redirect_url, code))
            st.query_params.clear()  # Clear code from URL
            if token:
                user_id, user_email = asyncio.run(get_email(client, token['access_token']))
                if user_email:
                    try:
                        user = auth.get_user_by_email(user_email)
                        if not user.email_verified:
                            st.warning("Please verify your email before logging in. Return to signup to complete verification.")
                            return None
                        st.session_state.email = user.email
                        st.session_state.username = user.uid
                        st.session_state.usermail = user.email
                        st.session_state.singout = True
                        st.session_state.singedout = True
                        return user.email
                    except exceptions.FirebaseError as e:
                        if 'not found' in str(e).lower():
                            user = auth.create_user(email=user_email)
                            # Generate verification link (display for manual verification)
                            action_code_settings = auth.ActionCodeSettings(
                                url=redirect_url,
                                handle_code_in_app=True
                            )
                            link = auth.generate_email_verification_link(user_email, action_code_settings)
                            st.info(f"New user created. Verification URL: ```{link}``` (Copy-paste into browser or check email.)")
                            return None
                        else:
                            raise
        return None
    except Exception as e:
        st.error(f"Google auth error: {e}")
        return None

def show_login_button():
    try:
        authorization_url = asyncio.run(client.get_authorization_url(
            redirect_url,
            scope=["email", "profile"],
            extras_params={"access_type": "offline"},
        ))
        st.markdown(f'<a href="{authorization_url}" target="_self">Login with Google</a>', unsafe_allow_html=True)
        get_logged_in_user_email()
    except Exception as e:
        st.error(f"Failed to generate auth URL: {e}")

# Initialize session state
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'usermail' not in st.session_state:
    st.session_state['usermail'] = ''
if 'singedout' not in st.session_state:
    st.session_state.singedout = False
if 'singout' not in st.session_state:
    st.session_state.singout = False
if 'email' not in st.session_state:
    st.session_state.email = ''

def login_callback():
    email = st.session_state.get('input_email', '').strip()
    if not email:
        st.warning("Please enter an email.")
        return
    try:
        user = auth.get_user_by_email(email)
        if not user.email_verified:
            st.warning("Email not verified. Please complete verification in the signup section.")
            return
        # Note: Admin SDK can't verify passwords; assume verified email + existence for demo
        st.success("Login successful!")
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        # Clear inputs
        st.session_state.input_email = ''
    except exceptions.FirebaseError as e:
        if 'not found' in str(e).lower():
            st.warning("User not found. Please sign up first.")
        else:
            st.error(f"Login error: {e}")
    except Exception as e:
        st.error(f"Unexpected login error: {e}")

def signup_callback():
    email = st.session_state.get('signup_email', '').strip()
    password = st.session_state.get('signup_password', '').strip()
    username = st.session_state.get('signup_username', '').strip()  # For display only
    if not all([email, password]):
        st.warning("Fill in email and password.")
        return
    try:
        user = auth.create_user(email=email, password=password)  # Auto-generates UID
        
        # Generate verification link (string, not object)
        action_code_settings = auth.ActionCodeSettings(
            url=redirect_url,
            handle_code_in_app=True
        )
        link = auth.generate_email_verification_link(email, action_code_settings)  # Returns str
        
        # Clear fields
        st.session_state.signup_email = ''
        st.session_state.signup_password = ''
        st.session_state.signup_username = ''
        st.success(f"Account created for {email}! UID: {user.uid}")
        
        # Show verification section
        verify_email_callback(email, link)
        
    except RefreshError as e:
        st.error(f"Auth failed (invalid service account): {e}. Generate a new key.")
    except Exception as e:
        st.error(f"Signup error: {e}")

def verify_email_callback(email, initial_link=None):
    """Handles email verification by applying the action code from the link."""
    st.subheader("🔐 Verify Your Email")
    st.info(f"Check your inbox (or spam) for a verification email from Firebase. Or use the link below.")
    
    if initial_link:
        st.code(initial_link)  # Display generated link as code block (copy-paste friendly)
    
    verification_url = st.text_input("Paste the verification URL from your email here:", key="verification_url")
    if st.button("Verify Email", key="verify_button"):
        if not verification_url:
            st.warning("Please paste the verification URL.")
            return
        try:
            # Extract oobCode from URL (standard Firebase format: ...oobCode=ABC123...)
            if 'oobCode' in verification_url:
                oob_code = verification_url.split('oobCode=')[1].split('&')[0]
                auth.apply_action_code(oob_code)
                st.success("Email verified successfully! You can now log in.")
                st.rerun()  # Refresh to enable login
            else:
                st.error("Invalid verification URL. Ensure it contains 'oobCode='.")
        except exceptions.InvalidActionCodeError:
            st.error("Invalid or expired verification code. Request a new one via email.")
        except Exception as e:
            st.error(f"Verification error: {e}")

def logout_callback():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ''
    st.session_state.usermail = ''
    st.session_state.email = ''

if not st.session_state['singedout']:
    choice = st.selectbox("Login/SignUp", ["Login", "SignUp"], key="choice_select")

    if choice == "Login":
        st.subheader("Login Section")
        st.text_input("Email", key='input_email')
        st.text_input("Password", type='password', key='input_password')
        if st.button("Login", key="login_button"):
            login_callback()
        st.markdown("Or use Google:")
        show_login_button()

    else:
        st.subheader("Create New Account")
        st.text_input("Email", key='signup_email')
        st.text_input("Password", type='password', key='signup_password')
        st.text_input("Username (display only)", key='signup_username')  # Optional
        if st.button("SignUp", key="signup_button"):
            signup_callback()
        st.markdown("Or use Google:")
        show_login_button()

if st.session_state.singout:
    st.subheader("Welcome Back!")
    st.text(f'Username: {st.session_state.username}')
    st.text(f'Email: {st.session_state.usermail}')
    if st.button("SignOut", key=f"signout_{st.session_state.usermail}"):  # Unique key
        logout_callback()
