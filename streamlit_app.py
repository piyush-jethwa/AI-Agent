import streamlit as st

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

#--------------------------------#
#      Sidebar Configuration     #
#--------------------------------#
def render_sidebar():
    """Render the sidebar and handle API key & model configuration.
    
    The sidebar allows users to:
    1. Select an LLM provider (OpenAI, GROQ, or Ollama)
    2. Choose or input a specific model
    3. Enter necessary API keys
    
    Returns:
        dict: Contains selected provider and model information
    """
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.write("")
        with st.expander("🤖 Model Selection", expanded=True):
            provider = "GROQ"
            
            if provider == "GROQ":
                model = st.selectbox(
                    "Select GROQ Model",
                    [
                        "llama-3.3-70b-versatile",
                        "Custom"
                    ],
                    index=0,
                    help="Choose from GROQ's available models. All these models support tool use and parallel tool use."
                )
                if model == "Custom":
                    model = st.text_input("Enter your custom GROQ model:", value="", help="Specify your custom model string")

        
        with st.expander("🔑 API Keys", expanded=True):
            # API keys are now handled via st.secrets for deployment
            st.info("🔑 API keys are configured via Streamlit secrets for deployment")

    # Apply theme
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stTextInput > div > div > input, .stTextArea > div > textarea {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border-color: #555555 !important;
        }
        .stSelectbox > div > div {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
    # Light theme is default, no additional styling needed


    return {
        "provider": provider,
        "model": model
    }
