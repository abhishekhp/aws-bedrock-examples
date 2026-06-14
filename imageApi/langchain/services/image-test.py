"""
Local smoke test for the LangChain image handler.

Run from the `services/` directory after installing the runtime deps:
    pip install -r requirements.txt
    python image-test.py

Requires AWS credentials with access to Bedrock + the target S3 bucket.
Set BUCKET_NAME (and optionally BEDROCK_REGION / ENHANCE_PROMPT) first.
"""
import json

from image import handler

event = {"body": json.dumps({"description": "A beautiful sunset over the ocean"})}

response = handler(event, {})
print(json.dumps(response, indent=2))
