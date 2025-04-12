import streamlit as st
import openai
import json
import base64
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

###################################
# Azure Cognitive Search Credentials (from secrets)
###################################
AZURE_SEARCH_ENDPOINT = st.secrets["azure"]["search_endpoint"]
AZURE_SEARCH_API_KEY = st.secrets["azure"]["search_api_key"]
AZURE_SEARCH_INDEX_NAME = st.secrets["azure"]["search_index_name"]

###################################
# Azure OpenAI Credentials (from secrets)
###################################
openai.api_type = st.secrets["openai"]["api_type"]
openai.api_base = st.secrets["openai"]["api_base"]
openai.api_version = st.secrets["openai"]["api_version"]
openai.api_key = st.secrets["openai"]["api_key"]

# Deployment ID (your Azure OpenAI model name, e.g. "o3-mini")
DEPLOYMENT_ID = st.secrets["openai"]["deployment_id"]

def query_azure_search(query, k=3):
    """
    Queries Azure Cognitive Search for the given query,
    returns up to k relevant documents concatenated as a context string.
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

def generate_answer(question, temperature=0.7, max_tokens=200):
    """
    1. Retrieve context from Azure Cognitive Search.
    2. Build a prompt with context + question.
    3. Call Azure OpenAI and return the answer.
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
        "The authority examines the MYTP submitted by the airport and publishes a consultation paper constituting the submissions by airport operator, justifications and rationale provided by airport operator for each regulatory block such as capex, opex, depreciation, fair rate of return, traffic and taxes, draft decisions or proposals or considerations by authority on the submissions made by the airport operator. "
        "Subsequent to publishing of consultation paper the authority invites comments and opinions from various stakeholders such as airlines, airline associations and other players in the aviation ecosystem. "
        "In addition to the same the authority also gives the opportunity for the airport operator to provide counter arguments against the draft decisions or proposals or considerations made by the authority in the consultation paper and the comments provided by the stakeholders. "
        "The authority examines the opinions from the stakeholders counter arguments from airport on the above said aspects to arrive at the final decision in the approved or final multi year tariff order for the airport operator. "
        "The consultation paper and tariff order for a particular control period consists of two major sections, namely, true up and projections of regulatory blocks such as capex, opex, depreciation, fair rate of return, traffic and taxes. "
        "The true up of regulatory blocks such as capex, opex, depreciation, fair rate of return, traffic and taxes mentioned in the tariff order or consultation paper for a specific control period refers to the reconciliation of actual numbers with respect to the approved numbers by the authority for the previous control period. "
        "The projections of regulatory blocks such as capex, opex, depreciation, fair rate of return, traffic and taxes mentioned in the tariff order or consultation paper for a specific control period refers to the projected or estimated numbers of regulatory blocks such as traffic, capex, opex, depreciation, fair rate of return and taxes for that particular control period. "
        "The user will ask questions to extract data from the consultation paper or tariff order prepared by the authority (Airport economic regulator authority of India) for various airport operators such as DIAL, CSMIA, etc. "
        "When the user asks a question pertaining to traffic submitted by an airport operator, it should only extract data from those tables which specifically has title mentioning that the traffic is submitted by the airport operator such as DIAL, MIAL, CSMIA, BIAL, etc. "
        "When the user asks a question pertaining to traffic proposed or approved by the authority, it should only extract data from those tables which specifically has title mentioning that the traffic is proposed or approved by the authority. "
        "When the user asks a question pertaining to traffic projected by an independent consultant, it should only extract data from those tables which specifically has title mentioning the name of the independent consultant. It should not take the data from those tables which mentions traffic submitted by the airport operator or traffic proposed or approved by authority. "
        "When the user asks a question pertaining to traffic, it can include multiple types of traffic such as passenger traffic (expressed in MPPA (million passenger per annum) or million), cargo traffic (expressed in MT or thousands metric tonnes) and ATM (also called departures and expressed in thousands or millions unit). Data for such subcategories should be separately extracted from the tables which specifically has title mentioning that the traffic is submitted by the airport operator or proposed by the authority. "
        "When the user asks a question pertaining to total passenger traffic, it should extract the data from the row which mentions the total traffic containing the summation of individual traffic figures of international passenger traffic and domestic passenger traffic. "
        "When the user asks a question pertaining to total cargo traffic, it should extract the data from the row which mentions the total traffic containing the summation of individual traffic figures of international cargo traffic and domestic cargo traffic. "
        "When the user asks a question pertaining to total ATM or departure traffic, it should extract the data from the row which mentions the total traffic containing the summation of individual traffic figures of international ATM or departures and domestic ATM or departures. "
        "When the user asks a question pertaining to the true up, the data should be extracted only from those tables which specifically has the title mentioning that the traffic is for true up. "
        "When the user asks a question pertaining to projected traffic figures or projections, the data should be extracted only from those tables which specifically has title mentioning projections or projected traffic figures or estimations. "
        "When the user asks a question pertaining to the projected traffic figures, the data should be extracted only from those tables which specifically has title mentioning that the traffic figures are projections. "
        "When the user asks year on year traffic figures, the data should be extracted based on the year given on the top row of the table for relevant traffic type. "
        "When the user asks CAGR of a traffic type (such as international, domestic, ATM), the data should be extracted if the same is explicitly mentioned and calculated. If the same is not mentioned the same can be calculated from the relevant table data. "
        "When answering, format your output as an HTML table with clear headers and rows. For example, your output should resemble:\n\n"
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"5\">\n"
        "  <thead>\n"
        "    <tr>\n"
        "      <th>Fiscal Year (FY end)</th>\n"
        "      <th>International ATM (Billable) (in '000s)</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <td>2025</td>\n"
        "      <td>59.31</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <td>2026</td>\n"
        "      <td>62.18</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <td>2027</td>\n"
        "      <td>65.52</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <td>2028</td>\n"
        "      <td>68.59</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <td>2029</td>\n"
        "      <td>71.33</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <td><strong>TOTAL</strong></td>\n"
        "      <td><strong>326.92</strong></td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n\n"
        "References:\n"
        "- Document: 17383225584994.pdf – “TRAFFIC PROJECTIONS FOR THE FOURTH CONTROL PERIOD”\n"
        "- Table: Table 179 – Traffic Projections submitted by DIAL for the Fourth Control Period\n"
        "Also ensure that data is only extracted from tables whose titles specify the source (e.g., 'submitted by DIAL', 'proposed by the Authority', etc.)."
    )

    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_ID,  # Use engine instead of deployment_id
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,  # Use max_tokens instead of max_completion_tokens
    )
    answer = response.choices[0].message["content"]
    return answer

def get_base64_image(image_path: str) -> str:
    """
    Reads an image file from the given path,
    returns the base64-encoded string (suitable for embedding in HTML).
    """
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

#############################
# Streamlit App with Conversation History
#############################
st.set_page_config(layout="wide")

# --- Specify the path to your local logo ---
logo_path = "bial_logo.png"  # update as needed
logo_base64 = get_base64_image(logo_path)

# --- Custom CSS for styling ---
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

# Sidebar: Settings
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

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Page Title / Header with local logo
st.markdown(
    f"""
    <div class="title-bar">
        <img src="data:image/png;base64,{logo_base64}" alt="BIAL Logo" style="height:50px; margin-right:20px;">
        <h1>Regulatory Fact Finder</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Section: Predefined Questions & Custom Input
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

# Submit Button: When clicked, generate and display the answer
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

# Conversation History: Display within an expander
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history, start=1):
            st.markdown(f"**Q{idx}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")



