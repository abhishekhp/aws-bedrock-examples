# Text-to-Image API (LangChain + AWS CDK)

A serverless text-to-image service. A user POSTs a description to an API Gateway
REST endpoint; a Lambda runs a **LangChain** pipeline that generates an image
with Amazon Bedrock (Stability AI), stores it in S3, and returns a presigned
download URL. All infrastructure is defined with the AWS CDK (Python).
**No Docker is required to build or deploy.**

## How the LangChain pipeline works

The handler in `services/image.py` is expressed as a LangChain LCEL chain:

```
{"description": str}
  --> (optional) ChatBedrock prompt enhancement   (prompt | ChatBedrock | StrOutputParser)
  --> Bedrock image-generation Runnable            (Stability model via invoke_model)
  --> persist-to-S3 + presign Runnable
  --> {"prompt", "key", "url"}
```

Note: `langchain-aws` provides native model classes for text (`ChatBedrock`)
and embeddings (`BedrockEmbeddings`), but Bedrock **image** models have no
built-in LangChain model class. The idiomatic approach — used here — is to wrap
the image `invoke_model` call as a LangChain `Runnable` and compose it with LCEL.
The optional prompt-enhancement step (off by default) uses Claude via
`ChatBedrock` to turn a short description into a richer image prompt; enable it
with the `ENHANCE_PROMPT=true` environment variable.

## Project layout

| Path | Purpose |
|------|---------|
| `services/image.py` | LangChain LCEL chain + API Gateway Lambda handler |
| `services/requirements.txt` | Runtime deps bundled into the Lambda (langchain-core, langchain-aws, boto3) |
| `image_generation_app/image_generation_app_stack.py` | CDK stack (S3, Lambda, API Gateway) + Docker-free local bundling |
| `app.py` | CDK app entry point |
| `requirements.txt` | CDK app dependencies |

## How dependency packaging works (no Docker)

`langchain-aws` pulls in compiled native packages (`pydantic-core`, `numpy`)
whose binaries must match the Lambda runtime (Amazon Linux / x86_64), not the
machine running `cdk deploy`. The stack defines a **CDK local bundler**
(`LocalPipBundling`) that runs:

```
pip install -r services/requirements.txt --target <asset> \
    --platform manylinux2014_x86_64 --implementation cp \
    --python-version 3.12 --only-binary=:all:
```

This downloads pre-built Linux/x86_64 wheels directly — nothing is compiled
locally and no container is launched. A Docker image is still declared as a
fallback, but it is only used if local bundling fails, so normal builds need no
Docker daemon.

## Prerequisites

- Python 3.12, Node.js, and the AWS CDK Toolkit (`npm i -g aws-cdk`)
- Network access to PyPI (to download the Lambda wheels at build time)
- Bedrock model access enabled for the Stability image model (and Claude, if you
  turn on prompt enhancement) in the configured region

## Setup

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Deploy

```
cdk synth        # builds the Lambda bundle locally (no Docker)
cdk deploy
```

The deploy output prints the API Gateway base URL. POST to `/image`:

```
POST https://<api-id>.execute-api.<region>.amazonaws.com/prod/image
Content-Type: application/json

{ "description": "your image description" }
```

Response:

```json
{ "url": "https://...presigned-s3-url...", "key": "1718390000.png" }
```

See `requests.http` for a ready-to-run request.

## Configuration (Lambda environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BUCKET_NAME` | (set by CDK) | Destination S3 bucket |
| `BEDROCK_REGION` | `us-west-2` | Region for Bedrock calls |
| `IMAGE_MODEL_ID` | `stability.stable-image-core-v1:1` | Image model |
| `ENHANCE_PROMPT` | `false` | Use Claude to enrich the prompt first |
| `TEXT_MODEL_ID` | `anthropic.claude-3-5-haiku-...` | Model for enhancement |
| `OUTPUT_FORMAT` | `png` | `png` or `jpeg` |
| `ASPECT_RATIO` | `1:1` | Stability aspect ratio |
| `PRESIGN_EXPIRY_SECONDS` | `3600` | Presigned URL lifetime |

## Tests

```
pip install -r requirements-dev.txt
pytest                 # CDK assertions; bundling disabled, no network/Docker needed
```

## Useful CDK commands

* `cdk ls`     list stacks
* `cdk synth`  emit the CloudFormation template
* `cdk deploy` deploy the stack
* `cdk diff`   diff against the deployed stack
