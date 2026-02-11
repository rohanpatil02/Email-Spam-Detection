import streamlit as st
import pandas as pd
from joblib import load
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2c5aa0;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e3a6f;
        box-shadow: 0 4px 12px rgba(44, 90, 160, 0.3);
    }
    .spam-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .ham-box {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2c5aa0;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #dee2e6;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #2c5aa0;
        box-shadow: 0 0 0 0.2rem rgba(44, 90, 160, 0.25);
    }
    h1 {
        color: #1e3a6f;
        font-weight: 700;
    }
    h2, h3 {
        color: #2c5aa0;
    }
    .success-message {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and dataset
@st.cache_resource
def load_model():
    try:
        model_data = load('spam_detector_complete.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('spam.csv')
        df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
        return df
    except Exception as e:
        st.warning(f"Dataset not found: {e}")
        return None

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load resources
model_data = load_model()
df = load_dataset()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mail.png", width=80)
    st.title("📧 Spam Detector")
    st.markdown("---")
    
    if model_data:
        st.markdown("### 🎯 Model Info")
        st.metric("Accuracy", f"{model_data['accuracy']:.2%}")
        st.metric("Features", f"{model_data['feature_count']:,}")
        st.markdown("---")
    
    st.markdown("### 📊 Navigation")
    page = st.radio("Choose a page:", ["🔍 Classify Message", "📈 Dashboard"])
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This app uses a Naive Bayes classifier to detect spam messages with high accuracy.")

# Main content
if model_data is None:
    st.error("❌ Model file not found! Please ensure 'spam_detector_complete.joblib' is in the same directory.")
    st.stop()

# PAGE 1: Classify Message
if page == "🔍 Classify Message":
    st.title("🔍 Spam Message Classifier")
    st.markdown("Enter a message below to check if it's spam or legitimate (ham).")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Message")
        message = st.text_area(
            "Type or paste your message here:",
            height=200,
            placeholder="Enter the message you want to classify...",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            classify_button = st.button("🔍 Classify", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        # Example messages
        st.markdown("### 💡 Try These Examples")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("📧 Legitimate Example", use_container_width=True):
                message = "Hey, are we still meeting for coffee tomorrow at 3pm?"
                st.rerun()
        
        with example_col2:
            if st.button("⚠️ Spam Example", use_container_width=True):
                message = "WINNER! You've won $1000000! Click here NOW to claim your prize! Limited time offer!"
                st.rerun()
        
        if classify_button and message:
            with st.spinner("🔄 Analyzing message..."):
                # Transform and predict
                vectorizer = model_data['vectorizer']
                model = model_data['model']
                
                message_cv = vectorizer.transform([message])
                prediction = model.predict(message_cv)[0]
                probabilities = model.predict_proba(message_cv)[0]
                
                spam_confidence = probabilities[1] * 100
                ham_confidence = probabilities[0] * 100
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': message[:50] + "..." if len(message) > 50 else message,
                    'prediction': 'Spam' if prediction == 1 else 'Ham',
                    'confidence': max(spam_confidence, ham_confidence)
                })
                
                # Display result
                st.markdown("---")
                st.markdown("### 🎯 Classification Result")
                
                if prediction == 1:  # Spam
                    st.markdown(f"""
                        <div class="spam-box">
                            <h2>⚠️ SPAM DETECTED</h2>
                            <p style="font-size: 18px; margin-top: 10px;">This message appears to be spam!</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:  # Ham
                    st.markdown(f"""
                        <div class="ham-box">
                            <h2>✅ LEGITIMATE MESSAGE</h2>
                            <p style="font-size: 18px; margin-top: 10px;">This message appears to be legitimate.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence scores
                st.markdown("### 📊 Confidence Scores")
                
                score_col1, score_col2 = st.columns(2)
                
                with score_col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #51cf66; margin-bottom: 10px;">🟢 Legitimate (Ham)</h4>
                            <h2 style="color: #2c5aa0; margin: 0;">{ham_confidence:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with score_col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ff6b6b; margin-bottom: 10px;">🔴 Spam</h4>
                            <h2 style="color: #2c5aa0; margin: 0;">{spam_confidence:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=spam_confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Spam Probability", 'font': {'size': 24, 'color': '#2c5aa0'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#2c5aa0"},
                        'bar': {'color': "#2c5aa0"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#dee2e6",
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 70], 'color': '#fff3cd'},
                            {'range': [70, 100], 'color': '#f8d7da'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📜 Recent Predictions")
        if st.session_state.prediction_history:
            for i, entry in enumerate(reversed(st.session_state.prediction_history[-5:])):
                prediction_type = entry['prediction']
                icon = "⚠️" if prediction_type == "Spam" else "✅"
                color = "#ff6b6b" if prediction_type == "Spam" else "#51cf66"
                
                st.markdown(f"""
                    <div style="background: white; padding: 12px; border-radius: 8px; 
                                margin-bottom: 10px; border-left: 4px solid {color};
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: {color};">{icon} {prediction_type}</span>
                            <span style="font-size: 12px; color: #6c757d;">{entry['confidence']:.1f}%</span>
                        </div>
                        <div style="font-size: 12px; color: #6c757d; margin-top: 5px;">
                            {entry['message']}
                        </div>
                        <div style="font-size: 10px; color: #adb5bd; margin-top: 5px;">
                            {entry['timestamp']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions yet. Classify a message to see history!")

# PAGE 2: Dashboard
elif page == "📈 Dashboard":
    st.title("📈 Model Statistics Dashboard")
    
    if df is not None:
        # Dataset overview
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Total Messages</h4>
                    <h2 style="color: #2c5aa0;">{len(df):,}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            spam_count = df['spam'].sum()
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Spam Messages</h4>
                    <h2 style="color: #ff6b6b;">{spam_count:,}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ham_count = len(df) - spam_count
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Ham Messages</h4>
                    <h2 style="color: #51cf66;">{ham_count:,}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            spam_percentage = (spam_count / len(df)) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Spam Rate</h4>
                    <h2 style="color: #2c5aa0;">{spam_percentage:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Distribution of Messages")
            
            # Pie chart
            labels = ['Legitimate (Ham)', 'Spam']
            values = [ham_count, spam_count]
            colors = ['#51cf66', '#ff6b6b']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 📏 Message Length Analysis")
            
            # Message length distribution
            df['message_length'] = df['Message'].str.len()
            
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=df[df['spam'] == 0]['message_length'],
                name='Ham',
                marker_color='#51cf66',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig_hist.add_trace(go.Histogram(
                x=df[df['spam'] == 1]['message_length'],
                name='Spam',
                marker_color='#ff6b6b',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig_hist.update_layout(
                barmode='overlay',
                xaxis_title='Message Length (characters)',
                yaxis_title='Frequency',
                legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                height=400,
                margin=dict(l=20, r=20, t=40, b=60),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#e9ecef'),
                yaxis=dict(gridcolor='#e9ecef')
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Model performance section
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Model Accuracy</h4>
                    <h2 style="color: #2c5aa0;">{model_data['accuracy']:.2%}</h2>
                    <p style="color: #6c757d; font-size: 14px; margin-top: 10px;">
                        Percentage of correctly classified messages
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with perf_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Feature Count</h4>
                    <h2 style="color: #2c5aa0;">{model_data['feature_count']:,}</h2>
                    <p style="color: #6c757d; font-size: 14px; margin-top: 10px;">
                        Unique words in vocabulary
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with perf_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #6c757d;">Algorithm</h4>
                    <h2 style="color: #2c5aa0; font-size: 24px;">Naive Bayes</h2>
                    <p style="color: #6c757d; font-size: 14px; margin-top: 10px;">
                        Multinomial variant
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Sample messages
        st.markdown("---")
        st.markdown("### 📝 Sample Messages from Dataset")
        
        sample_tab1, sample_tab2 = st.tabs(["✅ Ham Examples", "⚠️ Spam Examples"])
        
        with sample_tab1:
            ham_samples = df[df['spam'] == 0]['Message'].sample(min(5, len(df[df['spam'] == 0]))).tolist()
            for i, msg in enumerate(ham_samples, 1):
                st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; 
                                margin-bottom: 10px; border-left: 4px solid #51cf66;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <strong style="color: #51cf66;">Example {i}:</strong>
                        <p style="margin-top: 8px; color: #495057;">{msg}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with sample_tab2:
            spam_samples = df[df['spam'] == 1]['Message'].sample(min(5, len(df[df['spam'] == 1]))).tolist()
            for i, msg in enumerate(spam_samples, 1):
                st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; 
                                margin-bottom: 10px; border-left: 4px solid #ff6b6b;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <strong style="color: #ff6b6b;">Example {i}:</strong>
                        <p style="margin-top: 8px; color: #495057;">{msg}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Dataset not found. Dashboard statistics are unavailable.")
        st.info("To view full dashboard statistics, ensure 'spam.csv' is in the same directory as this app.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 20px;">
        <p>Built with ❤️ using Streamlit | Powered by Naive Bayes ML Algorithm</p>
        <p style="font-size: 12px;">© 2024 Spam Detector App</p>
    </div>
""", unsafe_allow_html=True)
