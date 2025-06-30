import streamlit as st
import streamlit_authenticator as stauth

# List of plain-text passwords
passwords = ["password123"]

# Hash each password one by one
hashed_passwords = [stauth.Hasher().hash(pw) for pw in passwords]
# Sample credentials
credentials = {
    "usernames": {
        "kapil": {
            "name": "Kapil Modi",
            "password": "abc",
        }
    }
}

# Initialize authenticator
authenticator = stauth.Authenticate(
    credentials,
    "mnit_auth_cookie",  # cookie name
    "abcdef",            # secret key
    cookie_expiry_days=1
)

login_result = authenticator.login(location="main")
if login_result is not None:
    name, auth_status, username = login_result
else:
    name = auth_status = username = None

# Handle login states
if auth_status is False:
    st.error("Username or password is incorrect.")
elif auth_status is None:
    st.warning("Please enter your username and password.")
elif auth_status:
    st.success(f"Welcome {name}!")
    st.write("Now you can access the main app.")