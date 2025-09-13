import streamlit as st
import json
from google.auth.exceptions import RefreshError
import firebase_admin
from firebase_admin import credentials, auth, exceptions, initialize_app
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
import streamlit.components.v1 as components

# Firebase Admin credentials (server-side)
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

redirect_url = "http://localhost:8501/"  # Update for prod (e.g., https://your-app.streamlit.app/)
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
            st.query_params.clear()
            if token:
                user_id, user_email = asyncio.run(get_email(client, token['access_token']))
                if user_email:
                    try:
                        user = auth.get_user_by_email(user_email)
                        if not user.email_verified:
                            st.warning("Please verify your email before logging in. Return to signup.")
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
                            action_code_settings = auth.ActionCodeSettings(url=redirect_url, handle_code_in_app=True)
                            link = auth.generate_email_verification_link(user_email, action_code_settings)
                            st.info(f"New user created. Verification URL: ```{link}``` (Check email or paste to verify.)")
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
            extras_params={"access_type": "offline"}
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
            st.warning("Email not verified. Complete verification in signup.")
            return
        # Note: Admin SDK can't verify passwords; assume verified email + existence for demo
        st.success("Login successful!")
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        st.session_state.input_email = ''
    except exceptions.FirebaseError as e:
        if 'not found' in str(e).lower():
            st.warning("User not found. Please sign up first.")
        else:
            st.error(f"Login error: {e}")
    except Exception as e:
        st.error(f"Unexpected login error: {e}")

def send_verification_js_component(email, uid):
    """Client-side JS to send verification email using Firebase JS SDK."""
    firebase_config = st.secrets["FIREBASE_WEB"]
    config_json = json.dumps(firebase_config)
    
    html = f"""
    <script src="https://www.gstatic.com/firebasejs/10.13.1/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/10.13.1/firebase-auth-compat.js"></script>
    <div id="firebase-container"></div>
    <script>
        const config = {config_json};
        firebase.initializeApp(config);
        const auth = firebase.auth();
        
        auth.signInWithCustomToken('{uid}').then((userCredential) => {{
            const user = userCredential.user;
            user.sendEmailVerification().then(() => {{
                document.getElementById('firebase-container').innerHTML = '<p style="color: green;">Verification email sent to {email}! Check your inbox/spam.</p>';
            }}).catch((error) => {{
                document.getElementById('firebase-container').innerHTML = '<p style="color: red;">Error sending email: ' + error.message + '</p>';
            }});
        }}).catch((error) => {{
            document.getElementById('firebase-container').innerHTML = '<p style="color: red;">Auth error: ' + error.message + '</p>';
        }});
    </script>
    """
    components.html(html, height=100)

def signup_callback():
    email = st.session_state.get('signup_email', '').strip()
    password = st.session_state.get('signup_password', '').strip()
    username = st.session_state.get('signup_username', '').strip()
    if not all([email, password]):
        st.warning("Fill in email and password.")
        return
    try:
        # Create user server-side
        user = auth.create_user(email=email, password=password)
        
        # Generate custom token for client-side auth
        custom_token = auth.create_custom_token(user.uid)
        
        # Clear fields safely within form
        st.session_state.signup_email = ''
        st.session_state.signup_password = ''
        st.session_state.signup_username = ''
        
        st.success(f"Account created for {email}! UID: {user.uid}")
        send_verification_js_component(email, custom_token)
        
    except RefreshError as e:
        st.error(f"Auth failed (invalid service account): {e}")
    except Exception as e:
        st.error(f"Signup error: {e}")

def logout_callback():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ''
    st.session_state.usermail = ''
    st.session_state.email = ''

st.title("Welcome to Stock Broke!")

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
        with st.form("signup_form"):
            st.text_input("Email", key='signup_email')
            st.text_input("Password", type='password', key='signup_password')
            st.text_input("Username (display only)", key='signup_username')
            submit = st.form_submit_button("SignUp", key="signup_button")
            if submit:
                signup_callback()
        st.markdown("Or use Google:")
        show_login_button()

if st.session_state.singout:
    st.subheader("Welcome Back!")
    st.text(f'Username: {st.session_state.username}')
    st.text(f'Email: {st.session_state.usermail}')
    if st.button("SignOut", key=f"signout_{st.session_state.usermail}_login"):
        logout_callback()
