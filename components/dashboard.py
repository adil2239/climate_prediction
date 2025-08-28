import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from enum import Enum
import pyrebase

# ================== AUTH CONFIG ==================
def init_firebase():
    firebaseConfig = {
        "apiKey": "YOUR_API_KEY",
        "authDomain": "YOUR_PROJECT.firebaseapp.com",
        "projectId": "YOUR_PROJECT",
        "storageBucket": "YOUR_PROJECT.appspot.com",
        "messagingSenderId": "SENDER_ID",
        "appId": "APP_ID",
        "databaseURL": ""
    }
    return pyrebase.initialize_app(firebaseConfig)

firebase = init_firebase()
auth = firebase.auth()

# ================== COLOR THEME ==================
COLOR_PALETTE = {
    "primary": "#2980b9",   # deep blue
    "success": "#27ae60",   # green
    "warning": "#f39c12",   # orange
    "danger": "#c0392b",    # red
    "dark": "#2c3e50",     # navy
    "light": "#ecf0f1",    # light gray
    "accent": "#8e44ad"    # purple
}

plt.style.use("seaborn-v0_8-darkgrid")

# ================== DATA LOADER ==================
def load_data():
    df = pd.read_csv("data/climate_data_2000_2024.csv")
    df['Year'] = df['Year'].astype(int)
    if 'Region' not in df.columns:
        regions = ['North', 'South', 'East', 'West', 'Central']
        df['Region'] = [regions[i % 5] for i in range(len(df))]
    if 'Anomaly' not in df.columns:
        anomalies = ['Heatwave', 'Flood', 'Drought', 'Cyclone', 'Normal']
        df['Anomaly'] = [anomalies[i % 5] for i in range(len(df))]
    return df

class DashboardPage(Enum):
    DASHBOARD = "Dashboard"
    LIVE_DATA = "Live Data"
    ANALYTICS = "Advanced Analytics"
    PREDICTIONS = "AI Predictions"
    REGIONS = "Regional Insights"
    FEEDBACK = "Feedback"

# ================== LOGIN PAGE ==================
def show_login_page():
    st.markdown(f"""
        <div style="background:{COLOR_PALETTE['primary']}; padding:15px; border-radius:12px; text-align:center">
            <h2 style="color:white">🔑 Login to Climate Insights</h2>
        </div>
    """, unsafe_allow_html=True)

    action = st.radio("Select Action", ["Login", "Register"], horizontal=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if action == "Login":
        if st.button("Login", use_container_width=True):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state['user'] = user
                st.session_state['authenticated'] = True
                st.success("✅ Logged in successfully!")
            except:
                st.error("❌ Invalid credentials")
    else:
        if st.button("Register", use_container_width=True):
            try:
                user = auth.create_user_with_email_and_password(email, password)
                st.success("✅ Account created. Please login now.")
            except:
                st.error("❌ Registration failed. Try again.")

# ================== MAIN ==================
def show_dashboard():
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        show_login_page()
        return

    df = load_data()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = DashboardPage.DASHBOARD.value

    st.markdown(f"""
        <div style="background:{COLOR_PALETTE['primary']}; padding:15px; border-radius:12px; text-align:center">
            <h1 style="color:white">🌍 Climate Insights</h1>
            <p style="color:{COLOR_PALETTE['light']}">Understanding Climate Through Data</p>
        </div>
    """, unsafe_allow_html=True)

    nav_options = [p.value for p in DashboardPage]
    page = st.selectbox("📌 Navigate", nav_options, index=nav_options.index(st.session_state.current_page))
    st.session_state.current_page = page

    st.subheader(f"{st.session_state.current_page}")

    if st.session_state.current_page == DashboardPage.DASHBOARD.value:
        show_overview_page(df)
    elif st.session_state.current_page == DashboardPage.LIVE_DATA.value:
        show_live_monitoring_page(df)
    elif st.session_state.current_page == DashboardPage.ANALYTICS.value:
        show_analytics_page(df)
    elif st.session_state.current_page == DashboardPage.PREDICTIONS.value:
        show_predictions_page(df)
    elif st.session_state.current_page == DashboardPage.REGIONS.value:
        show_regional_page(df)
    elif st.session_state.current_page == DashboardPage.FEEDBACK.value:
        show_feedback_page()

    st.markdown("""
        <hr>
        <p style="text-align:center; color:#7f8c8d">© 2025 Climate Insights • Built with Streamlit</p>
    """, unsafe_allow_html=True)

# ================== PAGES ==================
def show_overview_page(df):
    st.markdown("### 🌐 Quick Facts")
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Year", int(latest['Year']))
    col2.metric("CO₂", f"{latest['CO2(ppm)']:.1f} ppm")
    col3.metric("Temp", f"{latest['Temperature(C)']:.2f} °C")
    col4.metric("Entries", len(df))

    st.markdown("### 📊 Trends")
    chart1, chart2 = st.columns(2)
    with chart1:
        st.area_chart(df.set_index('Year')["CO2(ppm)"], color=COLOR_PALETTE['accent'])
    with chart2:
        st.area_chart(df.set_index('Year')["Temperature(C)"], color=COLOR_PALETTE['danger'])

    with st.expander("📋 Data Table"):
        st.dataframe(df)

def show_live_monitoring_page(df):
    st.markdown("### 📡 Live Simulation")
    placeholder = st.empty()
    if st.button("▶ Start Simulation"):
        streamed = pd.DataFrame()
        for year in df['Year'].unique():
            streamed = pd.concat([streamed, df[df['Year'] == year]])
            avg = streamed.groupby('Year').mean(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            avg[['CO2(ppm)', 'Temperature(C)']].plot(ax=ax, marker='o', color=[COLOR_PALETTE['primary'], COLOR_PALETTE['danger']])
            ax.set_title(f"Climate Metrics up to {year}")
            placeholder.pyplot(fig)
            time.sleep(0.25)

def show_analytics_page(df):
    st.markdown("### 🔎 Analytics")
    st.write("Correlation between numeric values")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("### ⚠ Anomalies")
    anomaly_counts = df['Anomaly'].value_counts()
    st.bar_chart(anomaly_counts, color=COLOR_PALETTE['warning'])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=df, x='Anomaly', y='Temperature(C)', ax=ax, palette="muted")
    st.pyplot(fig)

def show_predictions_page(df):
    st.markdown("### 🤖 Predictions")
    X = df[['Year']]
    y = df['CO2(ppm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
    st.metric("R²", f"{model.score(X_test, y_test):.2f}")

    future = pd.DataFrame({'Year': range(2025, 2031)})
    preds = model.predict(future)
    st.line_chart(pd.DataFrame({'Year': future['Year'], 'Predictions': preds}).set_index('Year'), color=COLOR_PALETTE['success'])

def show_regional_page(df):
    st.markdown("### 🌍 Regional Insights")
    region = st.selectbox("Select Region", df['Region'].unique())
    regional = df[df['Region'] == region]

    st.line_chart(regional.set_index('Year')["Temperature(C)"], color=COLOR_PALETTE['danger'])
    st.bar_chart(df.groupby('Region')["Rainfall(mm)"].sum(), color=COLOR_PALETTE['primary'])

def show_feedback_page():
    st.markdown("### 💬 Feedback")
    name = st.text_input("Your Name")
    email = st.text_input("Email")
    rating = st.slider("Rating", 1, 5, 3)
    comments = st.text_area("Comments")
    if st.button("Send"):
        if name and email:
            fb = pd.DataFrame([[name, email, rating, comments]])
            fb.to_csv("data/feedback.csv", mode='a', index=False, header=False)
            st.success("Feedback submitted!")
        else:
            st.error("Please fill name & email")

if __name__ == "__main__":
    st.set_page_config(page_title="Climate Insights Dashboard", layout="wide")
    show_dashboard()
