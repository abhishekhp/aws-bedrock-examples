import boto3
import json
import base64
from time import time
import os

AWS_REGION_BEDROCK = "us-west-2"
S3_BUCKET = "text-2-image-bucket-253138837329-us-east-1-an"

model_id = "stability.stable-image-core-v1:1"  # cheapest active text-to-image model

client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION_BEDROCK)
s3_client = boto3.client('s3')


def handler(event, context):
    body = json.loads(event["body"])
    description = body.get("description")
    if description:
        model_config = get_stability_config(description)
        response = client.invoke_model(
            body=model_config,
            modelId=model_id,
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        signed_url = save_image_to_s3(base64_image)
        return {
            "statusCode": 200,
            "body": json.dumps({"url": signed_url}),
        
        }


def save_image_to_s3(base64_image: str):
    image_file = base64.b64decode(base64_image)
    timestamp = int(time())
    image_name = str(timestamp) + '.jpg'

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=image_name,
        Body=image_file,
    )
    signed_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET, 'Key': image_name},
        ExpiresIn=3600
    )
    return signed_url


def get_titan_config(description: str):
    return json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": description,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": 8.0,
            },
        }
    )

def get_stability_config(description: str):
    return json.dumps({
    "prompt": description,
    "output_format": "png",
    "aspect_ratio": "1:1",   # options: 1:1, 16:9, 4:3, 3:2, 5:4, 2:3, 9:16
    "mode": "text-to-image"
    })