import json

from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

AWS_REGION_BEDROCK = "us-west-2"
MODEL_ID = "us.amazon.nova-lite-v1:0"

# Build the LangChain components once per container (reused across warm invocations).
llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION_BEDROCK,
    temperature=0,
    top_p=1,
    max_tokens=4096,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "From the text below, summarize the story in {points} points.\n\n"
            "Text: {text}",
        ),
    ]
)

# LCEL chain: prompt -> model -> plain string output.
summary_chain = prompt | llm | StrOutputParser()


def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text")
    query_params = event.get("queryStringParameters") or {}
    points = query_params.get("points")

    if text and points:
        result = get_summary(text, points)
        return {
            "statusCode": 200,
            "body": json.dumps({"summary": result}),
        }

    return {
        "statusCode": 400,
        "body": json.dumps({"error": "text and points required!"}),
    }


def get_summary(text: str, points: str) -> str:
    """Run the LangChain summarization chain and return the summary text."""
    return summary_chain.invoke({"text": text, "points": points})
