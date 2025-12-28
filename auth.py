import streamlit as st
import pandas as pd
import os

USER_FILE = "users.csv"

# -------------------------------
# Load users safely
# -------------------------------
def load_users():
    if not os.path.exists(USER_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USER_FILE, index=False)
        return df
    return pd.read_csv(USER_FILE)

# -------------------------------
# Save new user
# -------------------------------
def save_user(username, password):
    df = load_users()
    new_user = pd.DataFrame(
        [[username.strip(), password.strip()]],
        columns=["username", "password"]
    )
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)

# -------------------------------
# Login Page
# -------------------------------
def login_page():
    st.subheader("🔐 Login")

    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password").strip()

    if st.button("Login"):
        users = load_users()

        if users.empty:
            st.error("No users found. Please sign up first.")
            return

        valid_user = (
            (users["username"].astype(str).str.strip() == username) &
            (users["password"].astype(str).str.strip() == password)
        )

        if valid_user.any():
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid username or password ❌")

# -------------------------------
# Signup Page
# -------------------------------
def signup_page():
    st.subheader("📝 Sign Up")

    new_user = st.text_input("Create Username").strip()
    new_pass = st.text_input("Create Password", type="password").strip()

    if st.button("Register"):
        users = load_users()

        if new_user == "" or new_pass == "":
            st.error("Username and password cannot be empty")
            return

        if new_user in users["username"].astype(str).values:
            st.error("Username already exists")
        else:
            save_user(new_user, new_pass)
            st.success("Account created! Please login 🔑")

# -------------------------------
# Logout
# -------------------------------
def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()