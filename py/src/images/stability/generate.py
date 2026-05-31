import boto3
import json
import base64

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

model_id = "stability.stable-image-core-v1:1"  # cheapest active text-to-image model

stability_config = json.dumps({
    "prompt": "cat on a mat on a country hillside",
    "output_format": "png",
    "aspect_ratio": "1:1",   # options: 1:1, 16:9, 4:3, 3:2, 5:4, 2:3, 9:16
    "mode": "text-to-image"
})

response = client.invoke_model(
    body=stability_config,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]

image_data = base64.b64decode(base64_image)

file_path = "cat.png"
with open(file_path, "wb") as f:
    f.write(image_data)

print(f"Image saved to {file_path}")