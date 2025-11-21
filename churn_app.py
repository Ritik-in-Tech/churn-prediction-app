import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import time

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="ChurnGuard | Real-Time Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .high-risk {
        color: #dc3545;
        font-weight: bold;
    }
    .low-risk {
        color: #28a745;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATION ENGINE ---
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """
    Generates a synthetic dataset representing subscription users.
    Features: Tenure, MonthlyCharge, UsageFrequency, SupportCalls, LastLoginDays, SentimentScore
    Target: Churn (0 or 1)
    """
    np.random.seed(42)
    
    # Generate features
    tenure = np.random.randint(1, 72, n_samples)  # Months
    monthly_charges = np.random.normal(70, 30, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    
    usage_freq = np.random.randint(0, 30, n_samples) # Times used per month
    support_calls = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.04, 0.01])
    last_login_days = np.random.randint(0, 60, n_samples)
    sentiment_score = np.random.uniform(0, 10, n_samples) # Customer sentiment from survey (0=Bad, 10=Good)

    data = pd.DataFrame({
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'UsageFrequency': usage_freq,
        'SupportCalls': support_calls,
        'LastLoginDays': last_login_days,
        'SentimentScore': sentiment_score
    })

    # Generate Target (Churn) based on logical rules + noise
    # Higher churn probability if: Low tenure, High charges, Low usage, High support calls, Low sentiment
    churn_prob = (
        (data['MonthlyCharges'] / 150) * 0.3 +
        (data['SupportCalls'] / 5) * 0.4 +
        (data['LastLoginDays'] / 60) * 0.4 - 
        (data['Tenure'] / 72) * 0.3 -
        (data['SentimentScore'] / 10) * 0.4
    )
    
    # Add random noise
    churn_prob += np.random.normal(0, 0.1, n_samples)
    
    # Normalize to 0-1 (clipping for sigmoid-like behavior)
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
    
    # Threshold for binary classification
    data['Churn'] = (churn_prob > 0.55).astype(int)
    
    return data

# --- MODEL TRAINING ENGINE ---
@st.cache_resource
def train_model(df):
    """
    Trains a Random Forest model on the provided data.
    Returns the model, accuracy, and feature importance.
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    return model, acc, auc, X_train.columns

# --- UI COMPONENTS ---

def display_header():
    st.title("📉 ChurnGuard AI")
    st.markdown("**Real-Time Customer Churn Prediction System**")
    st.markdown("""
    This dashboard simulates a production ML pipeline. It uses behavioral logs and interaction data 
    to predict the likelihood of a customer cancelling their subscription.
    """)
    st.divider()

def sidebar_input_features():
    st.sidebar.header("User Simulator")
    st.sidebar.markdown("Adjust values to simulate a customer profile:")
    
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12, help="How long the user has been a customer.")
    monthly_charges = st.sidebar.slider("Monthly Bill ($)", 20, 150, 70)
    usage_freq = st.sidebar.slider("Usage Frequency (Logins/Month)", 0, 30, 5)
    support_calls = st.sidebar.selectbox("Support Calls (Last 30 Days)", [0, 1, 2, 3, 4, 5], index=0)
    last_login_days = st.sidebar.slider("Days Since Last Login", 0, 60, 3)
    sentiment_score = st.sidebar.slider("Sentiment Score (CSAT)", 0.0, 10.0, 5.0, help="0 = Angry, 10 = Happy")
    
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'UsageFrequency': [usage_freq],
        'SupportCalls': [support_calls],
        'LastLoginDays': [last_login_days],
        'SentimentScore': [sentiment_score]
    })
    
    return input_data

def plot_gauge(prob):
    """Creates a gauge chart for churn probability."""
    color = "green"
    if prob > 0.3: color = "orange"
    if prob > 0.7: color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#e6ffe6"},
                {'range': [30, 70], 'color': "#fff4e6"},
                {'range': [70, 100], 'color': "#ffe6e6"}],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def get_recommendation(prob, input_data):
    """Business logic rule engine for recommendations."""
    if prob < 0.3:
        return "✅ **Safe:** No immediate action needed. Standard engagement."
    elif prob < 0.7:
        reasons = []
        if input_data['SentimentScore'].values[0] < 5:
            reasons.append("Low Sentiment")
        if input_data['SupportCalls'].values[0] > 2:
            reasons.append("High Support Friction")
        
        return f"⚠️ **At Risk:** {', '.join(reasons)}. **Action:** Send 'We miss you' campaign or offer 10% discount."
    else:
        return "🚨 **High Risk:** Immediate intervention required! **Action:** Assign account manager for direct outreach + 30% loyalty discount."

# --- MAIN APP LOGIC ---

def main():
    display_header()
    
    # 1. Load & Train
    with st.spinner('Fetching data from Data Warehouse and Training Model...'):
        df = generate_synthetic_data()
        model, acc, auc, feature_names = train_model(df)
        # Simulate a brief delay for realism
        time.sleep(0.5)

    # 2. Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Current Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    col3.metric("Model Accuracy", f"{acc*100:.1f}%")
    col4.metric("Model AUC", f"{auc:.2f}")
    
    st.divider()

    # 3. Real-Time Prediction Section
    st.subheader("🔍 Single User Analysis")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        # Get input from sidebar
        input_data = sidebar_input_features()
        
        # Prediction
        prediction_prob = model.predict_proba(input_data)[0][1]
        prediction_class = "Churn" if prediction_prob > 0.5 else "Retain"
        
        st.plotly_chart(plot_gauge(prediction_prob), use_container_width=True)
    
    with c2:
        st.markdown("### Analysis Results")
        st.info(f"**Recommendation:** {get_recommendation(prediction_prob, input_data)}")
        
        # Explainability (Simple Feature Contribution)
        st.markdown("#### Why this score?")
        
        # Simple heuristic for explainability visualization
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', 
                         title="Global Model Feature Importance", color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # 4. Bulk Analysis / Dashboard View
    st.subheader("📊 High Risk Customer Monitor")
    st.markdown("Below is a live feed of customers identified as 'High Risk' by the model (Probability > 70%).")
    
    # Generate predictions for the whole dataset
    all_probs = model.predict_proba(df.drop('Churn', axis=1))[:, 1]
    df['Churn Probability'] = all_probs
    
    high_risk_users = df[df['Churn Probability'] > 0.7].sort_values(by='Churn Probability', ascending=False).head(10)
    
    # Styled dataframe
    st.dataframe(
        high_risk_users.style.background_gradient(subset=['Churn Probability'], cmap='Reds'),
        use_container_width=True
    )
    
    st.download_button(
        label="📥 Download Risk Report (CSV)",
        data=high_risk_users.to_csv().encode('utf-8'),
        file_name='high_risk_churn_report.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()
