import boto3
import json
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

AWS_REGION_BEDROCK = "us-west-2"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION_BEDROCK)

model = ChatBedrock(model_id="us.amazon.nova-lite-v1:0", client=bedrock)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that summarizes text into concise bullet points.",
        ),
        (
            "user",
            "From the text below, summarize the story in {points} points.\n\nText: {text}",
        ),
    ]
)

chain = template | model


def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text")
    points = event["queryStringParameters"]["points"]

    if text and points:
        response = chain.invoke({"text": text, "points": points})
        return {
            "statusCode": 200,
            "body": json.dumps({"summary": response.content}),
        }
    return {
        "statusCode": 400,
        "body": json.dumps({"error": "text and points required!"}),
    }