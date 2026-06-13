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
)
from constructs import Construct

LAMBDA_RUNTIME = aws_lambda.Runtime.PYTHON_3_11

# Force pip to fetch wheels built for the Lambda execution environment
# (Amazon Linux, x86_64, CPython 3.11) regardless of the machine running
# `cdk synth`. This is what lets us bundle dependencies locally — without
# Docker — even on macOS or Windows.
_PIP_PLATFORM_ARGS = [
    "--platform", "manylinux2014_x86_64",
    "--implementation", "cp",
    "--python-version", "3.11",
    "--only-binary=:all:",
]


@jsii.implements(ILocalBundling)
class _LocalPipBundling:
    """Installs the Lambda's dependencies on the host (no Docker required)."""

    def __init__(self, source_dir: str, requirements: str) -> None:
        self._source_dir = source_dir
        self._requirements = requirements

    def try_bundle(self, output_dir: str, *, image, **kwargs) -> bool:  # noqa: ARG002
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "-r", self._requirements,
                    *_PIP_PLATFORM_ARGS,
                    "--target", output_dir,
                ],
                check=True,
            )
            # Copy the handler source alongside the installed dependencies.
            shutil.copytree(self._source_dir, output_dir, dirs_exist_ok=True)
        except Exception:  # noqa: BLE001 - signal CDK to fall back to Docker
            return False
        return True


class PyStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Bundle the handler together with its LangChain dependencies.
        # `local` runs the pip install on the host (no Docker); the `image`
        # command is only used as a fallback if local bundling fails.
        summary_code = aws_lambda.Code.from_asset(
            "services",
            bundling=BundlingOptions(
                local=_LocalPipBundling("services", "services/requirements.txt"),
                image=LAMBDA_RUNTIME.bundling_image,
                command=[
                    "bash",
                    "-c",
                    "pip install -r requirements.txt -t /asset-output "
                    "&& cp -au . /asset-output",
                ],
            ),
        )

        summary_lambda = aws_lambda.Function(
            self,
            "Py-SummaryLambda",
            runtime=LAMBDA_RUNTIME,
            code=summary_code,
            handler="summary.handler",
            # LangChain imports add cold-start time and memory, so give the
            # function a little more headroom than the original.
            memory_size=512,
            timeout=Duration.seconds(60),
        )

        summary_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            effect=aws_iam.Effect.ALLOW,
            resources=["*"],
            actions=["bedrock:InvokeModel"]
        ))

        api = aws_apigateway.RestApi(self, "PY-SummaryApi")

        text_resource = api.root.add_resource("text")
        summary_integration = aws_apigateway.LambdaIntegration(summary_lambda)
        text_resource.add_method("POST", summary_integration)
