from fastapi import FastAPI
from routes import image, chat


app = FastAPI(title="🚗 Car Assistant Chatbot API")

app.include_router(image.image_router)
app.include_router(chat.chat_router)