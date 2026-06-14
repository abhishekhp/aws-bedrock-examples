"""
LangChain-based text-to-image Lambda handler.

Flow (expressed as a LangChain LCEL chain):

    {"description": str}
        --> (optional) ChatBedrock prompt-enhancement step          [RunnablePassthrough.assign]
        --> Bedrock image-generation Runnable (Stability model)      [RunnableLambda]
        --> persist to S3 + presign Runnable                         [RunnableLambda]
        --> {"prompt": str, "url": str, "key": str}

Why a custom Runnable for image generation?
    LangChain's `langchain-aws` package ships first-class model classes for
    *text/chat* (ChatBedrock) and *embeddings* (BedrockEmbeddings), but Bedrock
    image models (Stability / Titan) have no native LangChain "model" class.
    The idiomatic way to bring image generation into a LangChain pipeline is to
    wrap the `invoke_model` call as a `Runnable` / tool and compose it with LCEL,
    which is exactly what we do below. The optional prompt-enhancement step is a
    real LangChain sub-chain (prompt | ChatBedrock | StrOutputParser).
"""

import base64
import json
import logging
import os
from time import time
from typing import Any, Dict

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration (env-driven; sensible defaults preserve original behaviour)
# --------------------------------------------------------------------------- #
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-west-2")
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "stability.stable-image-core-v1:1")
# Model used only for the optional prompt-enhancement step:
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0")
# The CDK stack injects the generated bucket name here. Falls back to the
# original hard-coded value so the module still imports outside the stack.
S3_BUCKET = os.environ.get("BUCKET_NAME", "text-2-image-bucket-253138837329-us-east-1-an")
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "png")  # png | jpeg
ASPECT_RATIO = os.environ.get("ASPECT_RATIO", "1:1")
PRESIGN_EXPIRY = int(os.environ.get("PRESIGN_EXPIRY_SECONDS", "3600"))
# Use Claude to turn a short description into a richer image prompt.
# Off by default to preserve the original behaviour and avoid requiring
# Claude model access in Bedrock. Set ENHANCE_PROMPT=true to enable.
ENHANCE_PROMPT = os.environ.get("ENHANCE_PROMPT", "false").lower() == "true"

# --------------------------------------------------------------------------- #
# AWS clients (created once per warm container)
# --------------------------------------------------------------------------- #
bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
s3_client = boto3.client("s3")


# --------------------------------------------------------------------------- #
# Chain step 1 (optional): enhance the user's description with ChatBedrock
# --------------------------------------------------------------------------- #
def _build_prompt_enhancer():
    """Return a LangChain sub-chain: prompt | ChatBedrock | StrOutputParser."""
    # Imported lazily so the handler still works (image-only) even if the
    # text model isn't enabled in this account/region.
    from langchain_aws import ChatBedrock

    llm = ChatBedrock(
        model_id=TEXT_MODEL_ID,
        region_name=BEDROCK_REGION,
        model_kwargs={"max_tokens": 256, "temperature": 0.7},
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite short image descriptions into vivid, concrete "
                "text-to-image prompts. Reply with ONLY the rewritten prompt, "
                "no preamble, under 60 words.",
            ),
            ("human", "{description}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _resolve_prompt(payload: Dict[str, Any]) -> str:
    """Compute the final image prompt from the input description."""
    description = payload["description"]
    if not ENHANCE_PROMPT:
        return description
    try:
        enhanced = _PROMPT_ENHANCER.invoke({"description": description}).strip()
        logger.info("Enhanced prompt: %s", enhanced)
        return enhanced or description
    except Exception:  # never fail the request just because enhancement failed
        logger.exception("Prompt enhancement failed; using raw description")
        return description


# --------------------------------------------------------------------------- #
# Chain step 2: generate the image via the Bedrock Stability model
# --------------------------------------------------------------------------- #
def _stability_config(prompt: str) -> str:
    return json.dumps(
        {
            "prompt": prompt,
            "output_format": OUTPUT_FORMAT,
            "aspect_ratio": ASPECT_RATIO,  # 1:1, 16:9, 4:3, 3:2, 5:4, 2:3, 9:16
            "mode": "text-to-image",
        }
    )


def _generate_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload["prompt"]
    logger.info("Generating image for prompt: %s", prompt)
    response = bedrock_runtime.invoke_model(
        body=_stability_config(prompt),
        modelId=IMAGE_MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response["body"].read())
    base64_image = response_body["images"][0]
    return {**payload, "image_b64": base64_image}


# --------------------------------------------------------------------------- #
# Chain step 3: persist to S3 and return a presigned download URL
# --------------------------------------------------------------------------- #
def _persist_and_sign(payload: Dict[str, Any]) -> Dict[str, Any]:
    image_bytes = base64.b64decode(payload["image_b64"])
    key = f"{int(time())}.{OUTPUT_FORMAT}"
    content_type = "image/jpeg" if OUTPUT_FORMAT in ("jpg", "jpeg") else f"image/{OUTPUT_FORMAT}"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
    )
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    logger.info("Stored image s3://%s/%s", S3_BUCKET, key)
    return {"prompt": payload["prompt"], "key": key, "url": url}


# --------------------------------------------------------------------------- #
# Assemble the LCEL pipeline once per container
# --------------------------------------------------------------------------- #
_PROMPT_ENHANCER = _build_prompt_enhancer() if ENHANCE_PROMPT else None

image_chain = (
    RunnablePassthrough.assign(prompt=RunnableLambda(_resolve_prompt))
    | RunnableLambda(_generate_image)
    | RunnableLambda(_persist_and_sign)
)


def generate_image_url(description: str) -> Dict[str, Any]:
    """Public entrypoint: run the LangChain chain for a description."""
    return image_chain.invoke({"description": description})


# --------------------------------------------------------------------------- #
# API Gateway (REST) Lambda proxy handler
# --------------------------------------------------------------------------- #
def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except (TypeError, json.JSONDecodeError):
        return _response(400, {"error": "Request body must be valid JSON."})

    description = (body.get("description") or "").strip()
    if not description:
        return _response(400, {"error": "Missing required field 'description'."})

    try:
        result = generate_image_url(description)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Image generation failed")
        return _response(500, {"error": "Image generation failed.", "detail": str(exc)})

    return _response(200, {"url": result["url"], "key": result["key"]})


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
