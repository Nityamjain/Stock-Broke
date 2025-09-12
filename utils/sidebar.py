import streamlit as st

def render_sidebar_profile():
    """Render a consistent profile section in the sidebar if logged in.

    Shows name, email, and a sign out button. If not logged in, shows a CTA to login.
    """
    with st.sidebar:
        st.divider()
        user = st.session_state.get('user')
        logged_in = st.session_state.get('logged_in') is True and user is not None
        if logged_in:
            name = user.get('name') or (user.get('email').split('@')[0] if user.get('email') else 'User')
            email = user.get('email') or ''
            st.markdown('**Profile**')
            st.write(name)
            if email:
                st.caption(email)
            cols = st.columns([1, 1, 1])
            with cols[0]:
                st.page_link('pages/2_Dashboard.py', label='Dashboard')
            with cols[1]:
                st.page_link('pages/Watchlist.py', label='Watchlist')
            with cols[2]:
                st.page_link('app.py', label='Home')
            if st.button('Sign Out'):
                for k in ['logged_in', 'user', 'google_id_token']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.switch_page('pages/1_Login.py')
        else:
            st.markdown('**Welcome**')
            st.caption('Please sign in to personalize your experience')
            st.page_link('pages/1_Login.py', label='Login')


