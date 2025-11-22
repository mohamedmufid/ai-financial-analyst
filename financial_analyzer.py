import streamlit as st
import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen, Request

# --- Configuration for Gemini API ---
# In a real deployed app, the API key would be stored securely as a Streamlit Secret.
# For this example, we keep it as an empty string as per the environment rules.
API_KEY = "" 
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# --- Utility Functions ---

def fetch_gemini_response(user_query, system_prompt, max_retries=5):
    """
    Calls the Gemini API with exponential backoff and Google Search grounding.
    """
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "tools": [{"google_search": {}}],  # Enable Google Search for grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    # We must construct the API URL with the API key
    full_api_url = f"{API_URL_BASE}?key={API_KEY}"

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            # Prepare the request
            req = Request(
                full_api_url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers=headers, 
                method='POST'
            )
            
            # Make the API call
            with urlopen(req, timeout=10) as response:
                if response.status != 200:
                    raise HTTPError(full_api_url, response.status, f"HTTP Error: {response.status}", response.headers, None)

                result = json.loads(response.read().decode())
                
                # Extract and return text and sources
                candidate = result.get('candidates', [{}])[0]
                text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No analysis generated.')
                
                sources = []
                grounding_metadata = candidate.get('groundingMetadata')
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = [
                        {
                            'title': attr.get('web', {}).get('title'),
                            'uri': attr.get('web', {}).get('uri')
                        }
                        for attr in grounding_metadata['groundingAttributions']
                        if attr.get('web', {}).get('uri') and attr.get('web', {}).get('title')
                    ]
                
                return text, sources

        except HTTPError as e:
            # Handle rate limiting (status 429) or other temporary errors
            if attempt < max_retries - 1 and e.code in [429, 500, 503]:
                delay = 2 ** attempt  # Exponential backoff
                time.sleep(delay)
            else:
                st.error(f"Failed to fetch analysis after multiple retries. API Error: {e.reason}")
                return "Error fetching analysis.", []
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return "Error fetching analysis.", []

    return "Max retries reached. Could not complete the request.", []

# --- Streamlit UI ---

st.set_page_config(
    page_title="Gemini Financial Analyzer", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📈 Real-time Financial Market Analyzer")
st.markdown("Ask Gemini for a quick, grounded summary of the latest financial news or data analysis trends.")

# Define the input area
topic = st.text_input(
    "Enter a financial topic or stock symbol to analyze (e.g., 'S&P 500 performance this week', 'Tesla Q3 earnings summary'):",
    "Latest data analysis tools and trends in finance"
)

if st.button("Get Real-time Analysis", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic to analyze.")
    else:
        with st.spinner('Contacting market analysts (Gemini)...'):
            
            # Define system instructions for the LLM
            system_prompt = (
                "You are a world-class financial analyst and data scientist. "
                "Provide a concise, neutral, and single-paragraph summary of the key findings "
                "related to the user's query. Your summary must be based ONLY on the provided "
                "search results. If no relevant information is found, state that."
            )
            
            # Define the user query for the LLM
            user_query = f"Find the latest news and key findings for: {topic}"
            
            # Fetch the grounded response
            analysis, sources = fetch_gemini_response(user_query, system_prompt)
            
            st.subheader(f"Analysis for: {topic}")
            st.info(analysis)
            
            # Display sources if available
            if sources:
                st.subheader("Source Citations")
                source_markdown = ""
                for i, source in enumerate(sources):
                    source_markdown += f"{i+1}. [{source['title']}]({source['uri']})\n"
                st.markdown(source_markdown)
            else:
                st.markdown("_Note: No direct web sources were cited for this analysis._")

st.sidebar.header("Deployment Checklist")
st.sidebar.markdown("""
To deploy this app from GitHub to Streamlit Community Cloud:
1. **Commit** this `financial_analyzer.py` file to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) (or just search "Streamlit Cloud").
3. Click **'New app'** and connect your GitHub account.
4. Select the repository and the main branch.
5. Set the **Main file path** to `financial_analyzer.py`.
6. Click **'Deploy!'**
""")

st.sidebar.markdown("---")
st.sidebar.info("This app uses the Gemini API with Google Search grounding for real-time, evidence-based analysis.")