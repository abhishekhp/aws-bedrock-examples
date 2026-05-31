import boto3

client = boto3.client(service_name='bedrock', region_name="us-west-2")

response = client.list_foundation_models(byOutputModality='IMAGE')

for model in response['modelSummaries']:
    print(model['modelId'], "|", model['modelLifecycle']['status'])