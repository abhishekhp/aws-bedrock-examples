import boto3
import json
import base64
from PIL import Image
import io

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

def create_mask(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.new("RGB", img.size, "white")  # white = edit entire image
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_configuration(input_image: str, mask_image: str):
    return json.dumps({
        "prompt": "Make the cat black and blue",
        "negative_prompt": "bad quality, low res",
        "image": input_image,
        "mask": mask_image,
        "output_format": "png"
    })

with open("cat.png", "rb") as f:
    image_bytes = f.read()

base_image = base64.b64encode(image_bytes).decode("utf-8")
mask_image = create_mask(image_bytes)

response = client.invoke_model(
    body=get_configuration(base_image, mask_image),
    modelId="us.stability.stable-image-inpaint-v1:0",
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]
image_data = base64.b64decode(base64_image)

file_path = "cat_edited.png"
with open(file_path, "wb") as f:
    f.write(image_data)

print(f"Edited image saved to {file_path}")