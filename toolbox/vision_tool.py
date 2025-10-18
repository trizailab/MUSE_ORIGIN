import imghdr
import os
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

async def extract_image_content_by_gpt4o(
    image_path: str,
    query: str
):
    """
    Use vlm `gpt-4o` to recognize or understand local images.

    Args:
        image_path: Local image path.
        query: Query to the gpt-4o to get the image content.
    """
    path = Path(image_path)

    if not path.exists():
        yield {
            "data": f"Error: File not found at path `{image_path}`.",
            "instruction": ""
        }
        return

    img_type = imghdr.what(path)
    if img_type is None:
        yield {
            "data": f"Error: The file at `{image_path}` is not a valid image.",
            "instruction": ""
        }
        return

    try:
        image_bytes = path.read_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        yield {
            "data": f"Error: Failed to read/encode the image `{image_path}`. Details: {e}",
            "instruction": ""
        }
        return

    try:
        response = client.chat.completions.create(
            model=os.getenv("VISION_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1024
        )

        result = response.choices[0].message.content
        yield {
            "data": result,
            "instruction": ""
        }
    except Exception as e:
        yield {
            "data": f"Error: gpt-4o request failed. Details: {e}",
            "instruction": ""
        }