import streamlit as st
import openai
import base64
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

###################################
# Retrieve Credentials from Streamlit Secrets
###################################
# Add these in your Streamlit Cloud Secrets UI or in a local .streamlit/secrets.toml file.
# Example secrets.toml:
#
# [azure]
# search_endpoint = "https://your-search-endpoint.search.windows.net"
# search_api_key = "your-azure-search-api-key"
# search_index_name = "your-search-index-name"
#
# [openai]
# api_type = "azure"
# api_base = "https://your-openai-endpoint.openai.azure.com"
# api_version = "2024-12-01-preview"
# api_key = "your-azure-openai-key"
# deployment_id = "o3-mini"

AZURE_SEARCH_ENDPOINT = st.secrets["azure"]["search_endpoint"]
AZURE_SEARCH_API_KEY = st.secrets["azure"]["search_api_key"]
AZURE_SEARCH_INDEX_NAME = st.secrets["azure"]["search_index_name"]

openai.api_type = st.secrets["openai"]["api_type"]
openai.api_base = st.secrets["openai"]["api_base"]
openai.api_version = st.secrets["openai"]["api_version"]
openai.api_key = st.secrets["openai"]["api_key"]

DEPLOYMENT_ID = st.secrets["openai"]["deployment_id"]

###################################
# Function: Query Azure Cognitive Search
###################################
def query_azure_search(query, k=3):
    """
    Queries Azure Cognitive Search for the given query,
    concatenates up to k relevant documents as a context string.
    """
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    results = search_client.search(search_text=query, top=k)
    context = ""
    for doc in results:
        content = doc.get("content", "")
        context += content + "\n"
    return context.strip()

###################################
# Function: Generate Answer using Azure OpenAI
###################################
def generate_answer(question, temperature=0.7, max_tokens=200):
    """
    1. Retrieve context from Azure Cognitive Search.
    2. Build a prompt with context and question.
    3. Use the updated OpenAI API call with the correct parameters.
    """
    context = query_azure_search(question, k=3)
    if not context:
        context = "No relevant documents found."

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and include references if available. "
        "An airport operator in India submits a multiyear tariff proposal (MYTP) to Airports economic authority of India (AERA or authority) for every five years to obtain an approval for multi year tariff. "
        "Control Period means a period of five Tariff Years, during which the Multi Year Tariff Order and Tariff(s) as determined by the Authority pursuant to such order shall subsist. "
        "The authority examines various regulatory blocks to determine the tariff for the control period. Such regulatory blocks include capital expenditure (CAPEX), Operational expenditure (Opex), depreciation, fair rate of return, traffic and taxes. "
        "The authority examines the MYTP submitted by the airport and publishes a consultation paper constituting the submissions by airport operator, justifications and rationale provided by the airport operator for each regulatory block such as capex, opex, depreciation, fair rate of return, traffic and taxes, draft decisions or proposals or considerations by the authority on the submissions made by the airport operator. "
        "Subsequent to publishing the consultation paper the authority invites comments and opinions from various stakeholders such as airlines, airline associations and other players in the aviation ecosystem. "
        "In addition, the authority also gives the opportunity for the airport operator to provide counter arguments against the draft decisions or proposals or considerations made by the authority and the comments provided by the stakeholders. "
        "The consultation paper and tariff order for a particular control period consists of two major sections, namely, true up and projections of regulatory blocks such as capex, opex, depreciation, fair rate of return, traffic and taxes. "
        "When answering, format your output as an HTML table with clear headers and rows."
    )

    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_ID,          # Updated parameter: use 'engine' for Azure deployments
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,       # Pass the temperature as defined
        max_tokens=max_tokens          # Use 'max_tokens' instead of 'max_completion_tokens'
    )
    answer = response.choices[0].message["content"]
    return answer

###################################
# Function: Convert Image to Base64 for Embedding
###################################
def get_base64_image(image_path: str) -> str:
    """
    Reads an image file from the given path and returns a base64-encoded string.
    Suitable for embedding in HTML.
    """
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

###################################
# Streamlit App Setup and Layout Configuration
###################################
st.set_page_config(layout="wide")

# --- Local Logo Setup ---
logo_path = "bial_logo.png"  # Place your logo file in your repository
logo_base64 = get_base64_image(logo_path)

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    .title-bar {
        background-color: #f5f5f5;
        padding: 10px 20px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 6px solid #fa8072;
        display: flex;
        align-items: center;
    }
    .title-bar h1 {
        margin: 0;
        color: #333;
        font-size: 1.8rem;
    }
    .response-box {
        background-color: #f7f7f7;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    .conversation-history {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

###################################
# Sidebar: Settings for Model Configuration
###################################
st.sidebar.header("Settings")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Controls the creativity of the model's output"
)
max_tokens = st.sidebar.slider(
    "Max Content Length (Tokens)",
    min_value=50,
    max_value=4000,
    value=200,
    step=50,
    help="Controls the maximum tokens in the generated answer"
)

###################################
# Initialize Session State for Conversation History
###################################
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

###################################
# App Header: Title with Logo
###################################
st.markdown(
    f"""
    <div class="title-bar">
        <img src="data:image/png;base64,{logo_base64}" alt="BIAL Logo" style="height:50px; margin-right:20px;">
        <h1>Regulatory Fact Finder</h1>
    </div>
    """,
    unsafe_allow_html=True
)

###################################
# Input Section: Predefined Questions & Custom Question Input
###################################
st.write("Choose a predefined question or type your own question below:")
predefined_questions = [
    "what is passenger traffic submiited by DIAL for fourth control period?",
    "what is actual traffic submitted by DIAL for third control period?",
    "what is true up AIR TRAFFIC MOVEMENT submitted by DIAL for FY23?"
]
selected_predef = st.selectbox("Or pick one from the list:", ["(None)"] + predefined_questions)
st.write("**Or type your custom question below:**")
question_input = st.text_area("Your question:", key="question", height=100)
if selected_predef != "(None)":
    question_input = selected_predef

###################################
# Submit Button: Generate and Display Answer
###################################
if st.button("Submit"):
    if not question_input.strip():
        st.warning("Please enter a question or select a predefined one.")
    else:
        answer = generate_answer(question=question_input, temperature=temperature, max_tokens=max_tokens)
        st.markdown(
            f"""
            <div class="response-box">
                <strong>Response:</strong><br>
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        # Log the conversation in session state
        st.session_state.conversation_history.append({"question": question_input, "response": answer})

###################################
# Conversation History Display
###################################
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history, start=1):
            st.markdown(f"**Q{idx}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")


