import boto3
import json

AWS_REGION_BEDROCK = "us-west-2"

client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION_BEDROCK)


def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text")
    points = event["queryStringParameters"]["points"]
    if text and points:
        nova_config = get_nova_config(text, points)  # changed
        response = client.invoke_model(
            body=nova_config,
            modelId="us.amazon.nova-lite-v1:0",  # changed
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        result = response_body["output"]["message"]["content"][0]["text"]  # changed
        return {
            "statusCode": 200,
            "body": json.dumps({"summary": result}),
        }
    return {
        "statusCode": 400,
        "body": json.dumps({"error": "text and points required!"}),
    }


def get_nova_config(text: str, points: str):  # changed
    prompt = f"From the text below, summarize the story in {points} points.\n\nText: {text}"

    return json.dumps(
        {
            "messages": [  # changed
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            "inferenceConfig": {  # changed
                "maxTokens": 4096,  # changed
                "temperature": 0,
                "topP": 1,
            },
        }
    )