import boto3

client = boto3.client(service_name='bedrock', region_name="us-west-2")

response = client.list_inference_profiles()

for profile in response['inferenceProfileSummaries']:
    if 'stable-image-inpaint' in profile['inferenceProfileId']:
        print(profile['inferenceProfileId'])
        print(profile['inferenceProfileArn'])