# app.py

import streamlit as st
from controllers.b import ChatbotController

# -----------------------------------------------------------------------------
#                      Configure the Streamlit Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🚗 Car Assistant Chatbot", layout="centered")
st.title("🚗 Car Assistant Chatbot")

# -----------------------------------------------------------------------------
#                    Session State Initialization
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    # List to store the conversation messages (role/content/etc.)
    st.session_state.chat_history = []

if "car_details" not in st.session_state:
    # String to store the extracted car details from the image
    st.session_state.car_details = ""

if "image_processed" not in st.session_state:
    # Boolean flag to ensure the image is processed only once
    st.session_state.image_processed = False

# -----------------------------------------------------------------------------
#                      Instantiate the Chatbot Controller
# -----------------------------------------------------------------------------
chatbot = ChatbotController()

# -----------------------------------------------------------------------------
#                          Image Upload & Processing
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a car image 🚗", type=["jpg", "jpeg", "png"])
if uploaded_file and not st.session_state.image_processed:
    # 1) معالجة الصورة لمرة واحدة
    with st.spinner("Analyzing car details..."):
        try:
            car_details_text = chatbot.process_uploaded_image(uploaded_file)
            st.session_state.car_details = car_details_text
        except Exception as e:
            st.session_state.car_details = ""
            st.error(f"Error analyzing image: {str(e)}")

    # 2) إضافة رسالة (Assistant) تحتوي على الصورة وتفاصيلها إلى سجل المحادثة
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"**Car Details (from Image):**\n{st.session_state.car_details}",
        "image": uploaded_file,  # نخزن الصورة مباشرة
        "caption": "Uploaded Car Image"
    })

    # وضع علامة على أنه تم تحليل الصورة
    st.session_state.image_processed = True

# -----------------------------------------------------------------------------
#                        Display Existing Conversation
# -----------------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        # في حال احتوت الرسالة على صورة، نعرضها
        if "image" in message and message["image"] is not None:
            st.image(message["image"], caption=message.get("caption", ""), use_container_width=True)
        # نعرض النص إذا وجد
        if message.get("content"):
            st.markdown(message["content"])

# -----------------------------------------------------------------------------
#                           Chat Input Section
# -----------------------------------------------------------------------------
user_prompt = st.chat_input("Ask about cars!")
if user_prompt:
    # عرض رسالة المستخدم
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # تحويل تاريخ المحادثة لسلسلة نصية لتمريرها إلى الوكيل (Agent)
    conversation_history_str = "\n".join(
        f"{msg['role']}: {msg.get('content', '[No Content]')}"
        for msg in st.session_state.chat_history
    )

    # -----------------------------------------------------------------------------
    #                ReAct Agent Handling (Thought -> Action -> Observation -> Answer)
    # -----------------------------------------------------------------------------
    with st.spinner("Thinking..."):
        assistant_response = chatbot.react_agent(
            user_prompt=user_prompt,
            conversation_history=conversation_history_str,
            car_details=st.session_state.car_details
        )

    # عرض استجابة المساعد
    st.chat_message("assistant").markdown(assistant_response)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
