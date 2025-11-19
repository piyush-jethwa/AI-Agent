# Handle SQLite for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import streamlit as st
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
from src.components.sidebar import render_sidebar
from src.components.researcher import create_crew, create_research_tasks, run_research
from src.utils.output_handler import capture_output

#--------------------------------#
#         Streamlit App          #
#--------------------------------#
# Configure the page
st.set_page_config(
    page_title="CrewAI Research Assistant",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for research history
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Logo
st.image(
    "https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg",
    width=200
)
st.markdown("[Visit CrewAI](https://www.crewai.com/)")

# Main layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🔍 :red[CrewAI] Research Assistant", anchor=False)

# Research History Section
if st.session_state.research_history:
    st.divider()
    st.markdown("### 📚 Research History")
    selected_history = st.selectbox(
        "Select a previous research:",
        [f"{h['timestamp']} - {h['topic'][:50]}..." for h in st.session_state.research_history]
    )
    if st.button("Load Selected Research"):
        idx = [f"{h['timestamp']} - {h['topic'][:50]}..." for h in st.session_state.research_history].index(selected_history)
        st.markdown(st.session_state.research_history[idx]['result'])

# Render sidebar and get selection (provider and model)
selection = render_sidebar()

# Check API keys from st.secrets for deployment
if not st.secrets.get("GROQ_API_KEY"):
    st.warning("⚠️ GROQ API key not found in Streamlit secrets")
    st.stop()

if not st.secrets.get("EXA_API_KEY"):
    st.warning("⚠️ EXA API key not found in Streamlit secrets")
    st.stop()

# Create two columns for the input section
input_col1, input_col2, input_col3 = st.columns([1, 3, 1])
with input_col2:
    task_description = st.text_area(
        "What would you like to research?",
        value="Research the latest AI Agent",
        height=68
    )

    # Voice input button
    if st.button("🎤 Voice Input"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak your research query.")
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                task_description = text
                st.success(f"Recognized: {text}")
            except sr.UnknownValueError:
                st.error("Could not understand audio.")
            except sr.RequestError:
                st.error("Speech recognition service unavailable.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

col1, col2, col3 = st.columns([1, 0.5, 1])
with col2:
    start_research = st.button("🚀 Start Research", use_container_width=False, type="primary")

if start_research:
    with st.status("🤖 Researching...", expanded=True) as status:
        try:
            # Create persistent container for process output with fixed height.
            process_container = st.container(height=300, border=True)
            output_container = process_container.container()

            # Single output capture context.
            with capture_output(output_container):
                crew = create_crew(selection)
                tasks = create_research_tasks(crew, task_description)
                result = run_research(crew, tasks)
                status.update(label="✅ Research completed!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Error occurred", state="error")
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    # Convert CrewOutput to string for display and download
    result_text = str(result)

    # Add to research history
    st.session_state.research_history.append({
        'topic': task_description,
        'result': result_text,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Display the final result
    st.markdown(result_text)

    # Data Visualizations
    st.divider()
    st.markdown("### 📊 Data Visualizations")
    viz_col1, viz_col2 = st.columns(2)

    # Extract numerical data from result_text (simple regex for percentages, numbers)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', result_text)
    numbers = [float(n.rstrip('%')) for n in numbers if n]

    if numbers:
        with viz_col1:
            fig = px.histogram(numbers, nbins=10, title="Distribution of Numerical Data")
            st.plotly_chart(fig, use_container_width=True)

        with viz_col2:
            fig2 = go.Figure(data=[go.Pie(labels=[f"Value {i+1}" for i in range(len(numbers[:5]))], values=numbers[:5])])
            fig2.update_layout(title="Top 5 Numerical Values")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No numerical data found in the research report for visualization.")

    # Create download buttons
    st.divider()
    download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
    with download_col2:
        st.markdown("### 📥 Download Research Report")

        # Download as Markdown
        st.download_button(
            label="Download as Markdown",
            data=result_text,
            file_name="research_report.md",
            mime="text/markdown",
            help="Download the research report in Markdown format"
        )

        # Download as PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in result_text.split('\n'):
            pdf.cell(200, 10, txt=line, ln=True)
        pdf_output = pdf.output(dest='S').encode('latin-1')

        st.download_button(
            label="Download as PDF",
            data=pdf_output,
            file_name="research_report.pdf",
            mime="application/pdf",
            help="Download the research report in PDF format"
        )

# Add footer
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.caption("Made with ❤️ using [CrewAI](https://crewai.com), [Exa](https://exa.ai) and [Streamlit](https://streamlit.io)")
