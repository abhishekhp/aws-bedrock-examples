import aws_cdk as core
import aws_cdk.assertions as assertions

from text_summary_app.py_stack import PyStack


def _template():
    # Disable asset bundling so the stack can be synthesized in unit tests
    # without Docker (the LangChain deps are pip-installed during real synth).
    app = core.App(context={"aws:cdk:bundling-stacks": []})
    stack = PyStack(app, "py")
    return assertions.Template.from_stack(stack)


def test_summary_lambda_created():
    template = _template()
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "summary.handler",
            "Runtime": "python3.11",
            "Timeout": 60,
            "MemorySize": 512,
        },
    )


def test_rest_api_created():
    template = _template()
    template.resource_count_is("AWS::ApiGateway::RestApi", 1)
    # The /text resource exposes a POST method.
    template.has_resource_properties(
        "AWS::ApiGateway::Method",
        {"HttpMethod": "POST"},
    )


def test_bedrock_invoke_permission_present():
    template = _template()
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": "bedrock:InvokeModel",
                                "Effect": "Allow",
                            }
                        )
                    ]
                )
            }
        },
    )
