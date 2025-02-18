import streamlit as st
from controllers import SQL_AgentController
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.PromptTemplate import get_prompt_template



# Load settings and the prompt template
settings = get_settings()
prompt_template = get_prompt_template()


# Set up LLM/Vision Clients
llm_provider_factory = LLMProviderFactory(settings)

# Text generation client
text_generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
text_generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

# Vision client (for image processing)
vision_client =llm_provider_factory.create(provider=settings.VISION_BACKEND)
vision_client.set_vision_model(model_id=settings.VISION_MODEL_ID)

# Text classification client
text_generation_client_classification = llm_provider_factory.create(provider=settings.CLASSIFICATION_BACKEND)
text_generation_client_classification.set_generation_model(model_id = settings.CLASSIFICATION_MODEL_ID)  

# Text generation SQL  
text_generation_client_sql =  llm_provider_factory.create(provider=settings.SQL_BACKEND) 
text_generation_client_sql.set_generation_model(model_id = settings.SQL_MODEL_ID)
llm_sql = text_generation_client_sql.LLM_CHAT()

# Initialize the SQL Agent Controller
sql_agent = SQL_AgentController(llm_sql)


# Configure the Streamlit page

st.set_page_config(page_title="🚗 Car Assistant Chatbot", layout="centered")

# Initialize session state for chat history and image-derived car details
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "car_details" not in st.session_state:
    st.session_state.car_details = ""

st.title("🚗 Car Assistant Chatbot")

def decide_mode_llm(user_query: str) -> str:
    """
    Classify the user's query into one of two modes: 'sql' or 'chat'.
    """
    try:
        classification_raw = text_generation_client_classification.generate_text(
            prompt_template.get_classification_prompt(user_query)
        )
        classification = classification_raw.strip().lower()
        if "sql" in classification:
            return "sql"
        else:
            return "chat"
    except Exception as e:
        st.error(f"Error classifying query: {str(e)}")
        return "chat"

def process_uploaded_image(uploaded_file):
    """
    Process the uploaded car image to extract details using the vision client.
    """
    with st.spinner("Analyzing car details..."):
        try:
            car_details = vision_client.vision_to_text(uploaded_file)
            st.session_state.car_details = car_details
            st.write("**Car Details (from Image):**", car_details)
        except Exception as e:
            st.session_state.car_details = ""
            st.error(f"Error analyzing image: {str(e)}")

def handle_sql_mode(user_prompt: str) -> str:
    """
    Process SQL-related queries.
    """
    with st.spinner("Processing SQL query..."):
        try:
            assistant_response = sql_agent.chat_agent_with_sql(user_prompt)
            return assistant_response
        except Exception as e:
            return f"Error generating response: {str(e)}"

def handle_normal_chat_mode(user_prompt: str) -> str:
    """
    Process normal chat queries, incorporating the entire conversation context.
    """
    # Combine the current chat history with the new query
    conversation_history = "\n".join([message["content"] for message in st.session_state.chat_history])
    
    # Include image details if available
    if st.session_state.car_details:
        combined_prompt = (
            f"{conversation_history}\nUser's new question: {user_prompt}\n"
            f"Additional context from image: {st.session_state.car_details}"
        )
    else:
        combined_prompt = f"{conversation_history}\nUser's new question: {user_prompt}"
    
    query = prompt_template.text_propt_user(combined_prompt)
    with st.spinner("Processing normal chat..."):
        try:
            assistant_response = text_generation_client.generate_text(query)
            return assistant_response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Image Upload Section
uploaded_file = st.file_uploader("Upload a car image 🚗", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Car Image", use_container_width=True)
    process_uploaded_image(uploaded_file)

# Display Existing Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Section
user_prompt = st.chat_input("Ask about cars!")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    # Append the user's query to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # Determine processing mode based on the user's query
    mode = decide_mode_llm(user_prompt)
    
    if mode == "sql":
        assistant_response = handle_sql_mode(user_prompt)
    else:
        assistant_response = handle_normal_chat_mode(user_prompt)
    
    # Display the assistant's response and add it to the chat history
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
