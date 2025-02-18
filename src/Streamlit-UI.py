from controllers import RAGController
from controllers import SQL_AgentController
import streamlit as st
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.PromptTemplate import get_prompt_template
import re


rag =RAGController()
sql_agent =SQL_AgentController()

vectordb_info =rag.index_into_vector_db()

# Setup: Environment and Clients

prompt_template = get_prompt_template()
settings = get_settings()

# Initialize the LLM provider
llm_provider_factory = LLMProviderFactory(settings)
# Text generation client
text_generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
text_generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)
# vision client
vision_client = llm_provider_factory.create(provider=settings.VISION_BACKEND)
vision_client.set_vision_model(model_id=settings.VISION_MODEL_ID)

# Configure Streamlit page
st.set_page_config(
    page_title="🚗 Car Assistant Chatbot",
    layout="centered"
)


chat_type = st.sidebar.selectbox(
        'Choose Chat type ',
        ['RAG', 'chat' , 'sql']
    )
if chat_type == "chat" :
    # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "car_details" not in st.session_state:
        st.session_state.car_details = ""
        
    # Streamlit App UI   
    # App title
    st.title("🚗 Car Assistant Chatbot")

    # Upload and Analyze Car Image
    uploaded_file = st.file_uploader("Upload a car image 🚗", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Car Image", use_container_width=True)

        # Generate car details from the image
        with st.spinner("Analyzing car details..."):
            try:
                car_details = vision_client.vision_to_text(uploaded_file)
                st.session_state.car_details = car_details
                st.write("**Car Details (from Image):**", car_details)
            except Exception as e:
                st.session_state.car_details = ""
                st.error(f"Error analyzing the image: {str(e)}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 

    # Chatbot Section
    user_prompt = st.chat_input("Ask me about cars...")
    if user_prompt:
        # Combine user input with car details from the image
        if st.session_state.car_details:
            combined_prompt = f"{user_prompt}\nAdditional context from image: {st.session_state.car_details}"
        else:
            combined_prompt = user_prompt
            
        # Generate the query using the prompt template
        query =prompt_template.text_propt_user(combined_prompt)


        # Add user message to the chat history
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        # Generate AI response
        
        with st.spinner("Processing your query..."):
            try:
                assistant_response = text_generation_client.generate_text(query)
            except Exception as e:
                assistant_response = f"Error generating response: {str(e)}"
                
                
        if settings.GENERATION_MODEL_ID  == "deepseek-r1-distill-llama-70b":
            cleaned_text = re.sub(r"<think>.*?</think>", "", assistant_response, flags=re.DOTALL)
            assistant_response = cleaned_text.strip()


        # Add assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the AI's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

elif chat_type == "RAG":
        # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "car_details" not in st.session_state:
        st.session_state.car_details = ""
        
    # Streamlit App UI   
    # App title
    st.title("🚗 Car Assistant Chatbot")

    
    

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 

    # Chatbot Section
    user_prompt = st.chat_input("Ask me about cars...")
    if user_prompt:

        results = rag.search_vector_db_collection(user_prompt)
        prompt = f"User's question: {user_prompt} \n\n Search results:\n {results}"
        # Generate the query using the prompt template
       
        
        # Add user message to the chat history
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        # Generate AI response
        with st.spinner("Processing your query..."):
            try:
                assistant_response = text_generation_client.generate_text(prompt ,type_chat=chat_type)
            except Exception as e:
                assistant_response = f"Error generating response: {str(e)}"
                
                
        if settings.GENERATION_MODEL_ID  == "deepseek-r1-distill-llama-70b":
            cleaned_text = re.sub(r"<think>.*?</think>", "", assistant_response, flags=re.DOTALL)
            assistant_response = cleaned_text.strip()


        # Add assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the AI's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

elif chat_type == "sql":

        # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "car_details" not in st.session_state:
        st.session_state.car_details = ""
        
    # Streamlit App UI   
    # App title
    st.title("🚗 Car Assistant Chatbot")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 

    # Chatbot Section
    user_prompt = st.chat_input("Ask me about cars...")
    if user_prompt:
        # Add user message to the chat history
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        # Generate AI response
        with st.spinner("Processing your query..."):
            try:
                assistant_response = sql_agent.chat_agent_with_sql_scratch(user_prompt)
            except Exception as e:
                assistant_response = f"Error generating response: {str(e)}"
                
                

        # Add assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the AI's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    