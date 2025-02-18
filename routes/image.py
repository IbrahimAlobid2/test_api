from fastapi import APIRouter, UploadFile, File, HTTPException
from models import ImageUploadResponse
from controllers import ChatbotController
import uuid
from io import BytesIO

image_router = APIRouter()
chatbot = ChatbotController()

@image_router.post("/image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    This endpoint handles image uploads. It expects a single file of type JPG, JPEG, or PNG.
    Upon receiving the file, the content is read into memory to avoid saving to disk.
    The ChatbotController is then used to process the image and extract any relevant details.

    Returns:
        ImageUploadResponse: An object containing details extracted from the image.
    """
    allowed_extensions = {"jpg", "jpeg", "png"}
    filename_parts = file.filename.split(".")

    if len(filename_parts) < 2 or filename_parts[-1].lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, JPEG, or PNG is allowed."
        )

    # Generate a new filename to avoid conflicts
    new_filename = f"{uuid.uuid4()}.jpg"
    file.filename = new_filename

    # Read the file contents into memory
    contents = await file.read()
    file.file = BytesIO(contents)

    # Process the uploaded image using the vision client in ChatbotController
    car_details = chatbot.process_uploaded_image(file)

    return ImageUploadResponse(car_details=car_details)
