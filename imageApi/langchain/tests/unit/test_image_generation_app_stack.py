import aws_cdk as core
import aws_cdk.assertions as assertions

from image_generation_app.image_generation_app_stack import ImageGenerationAppStack


def _template():
    # Disable asset bundling so the stack synthesizes quickly in CI without
    # running pip (or Docker). Fine for resource-shape assertions.
    app = core.App(context={"aws:cdk:bundling-stacks": []})
    stack = ImageGenerationAppStack(app, "test")
    return assertions.Template.from_stack(stack)


def test_creates_s3_bucket():
    _template().resource_count_is("AWS::S3::Bucket", 1)


def test_creates_lambda_with_correct_handler_and_runtime():
    _template().has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "image.handler",
            "Runtime": "python3.12",
            "Timeout": 60,
            "MemorySize": 1024,
        },
    )


def test_creates_rest_api_with_image_post_method():
    template = _template()
    template.resource_count_is("AWS::ApiGateway::RestApi", 1)
    template.has_resource_properties(
        "AWS::ApiGateway::Method", {"HttpMethod": "POST"}
    )
