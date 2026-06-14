import os
import shutil
import subprocess
import sys

import jsii
from aws_cdk import (
    BundlingOptions,
    Duration,
    ILocalBundling,
    Stack,
    aws_apigateway,
    aws_iam,
    aws_lambda,
    aws_s3,
)
from constructs import Construct

# Directory holding the Lambda source + its requirements.txt.
SERVICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "services")

# Must match the Lambda runtime below. Used to pull matching manylinux wheels.
LAMBDA_PYTHON_VERSION = "3.12"
LAMBDA_PLATFORM = "manylinux2014_x86_64"  # x86_64 Lambda runtime


@jsii.implements(ILocalBundling)
class LocalPipBundling:
    """Bundle the Lambda WITHOUT Docker.

    Runs `pip install` on the host but pulls pre-built Linux/x86_64 wheels
    (``--platform``/``--only-binary``) so the compiled deps in langchain-aws
    (pydantic-core, numpy, ...) match the Lambda runtime regardless of the
    developer's OS/architecture. Nothing is compiled locally and no container
    is needed. If anything goes wrong this returns False and CDK transparently
    falls back to the Docker image declared in BundlingOptions.
    """

    def try_bundle(self, output_dir: str, *, image=None, **kwargs) -> bool:
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "-r", os.path.join(SERVICES_DIR, "requirements.txt"),
                    "--target", output_dir,
                    "--platform", LAMBDA_PLATFORM,
                    "--implementation", "cp",
                    "--python-version", LAMBDA_PYTHON_VERSION,
                    "--only-binary=:all:",
                    "--upgrade",
                ],
                check=True,
            )
            # Ship the handler source alongside the installed dependencies.
            shutil.copy2(
                os.path.join(SERVICES_DIR, "image.py"),
                os.path.join(output_dir, "image.py"),
            )
            return True
        except Exception as exc:  # noqa: BLE001 -> fall back to Docker bundling
            print(f"Local bundling failed ({exc}); falling back to Docker.")
            return False


class ImageGenerationAppStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        images_bucket = aws_s3.Bucket(self, "ImagesBucket")

        lambda_runtime = aws_lambda.Runtime.PYTHON_3_12

        images_lambda = aws_lambda.Function(
            self,
            "ImageLambda",
            runtime=lambda_runtime,
            handler="image.handler",
            code=aws_lambda.Code.from_asset(
                "services",
                bundling=BundlingOptions(
                    # Primary path: local, Docker-free wheel install.
                    local=LocalPipBundling(),
                    # Fallback only (used if local bundling returns False);
                    # never invoked while local bundling succeeds, so Docker
                    # is not required for normal deploys.
                    image=lambda_runtime.bundling_image,
                    command=[
                        "bash", "-c",
                        "pip install -r requirements.txt -t /asset-output "
                        "&& cp -au image.py /asset-output",
                    ],
                ),
            ),
            # LangChain + numpy + pydantic-core need more memory/time than the
            # original direct-boto3 handler (cold-start import + image gen).
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment={
                "BUCKET_NAME": images_bucket.bucket_name,
                "BEDROCK_REGION": "us-west-2",
                "IMAGE_MODEL_ID": "stability.stable-image-core-v1:1",
                # Set "true" to enable the ChatBedrock prompt-enhancement step
                # (requires the Claude text model enabled in Bedrock).
                "ENHANCE_PROMPT": "false",
                "TEXT_MODEL_ID": "anthropic.claude-3-5-haiku-20241022-v1:0",
            },
        )

        images_bucket.grant_read_write(images_lambda)

        images_lambda.add_to_role_policy(
            aws_iam.PolicyStatement(
                effect=aws_iam.Effect.ALLOW,
                resources=["*"],
                actions=["bedrock:InvokeModel"],
            )
        )

        api = aws_apigateway.RestApi(self, "ImageApi")
        image_resource = api.root.add_resource("image")
        image_integration = aws_apigateway.LambdaIntegration(images_lambda)
        image_resource.add_method("POST", image_integration)
