import boto3
import json

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

history = []

def build_prompt():
    prompt = "<|begin_of_text|>"
    for msg in history:
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def get_configuration():
    return json.dumps({
        "prompt": build_prompt(),
        "max_gen_len": 512,
        "temperature": 0,
        "top_p": 1
    })

print("Bot: Hello! I am a chatbot. I can help you with anything you want to talk about.")

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    history.append({
        "role": "user",
        "content": user_input
    })

    response = client.invoke_model(
        body=get_configuration(),
        modelId="meta.llama3-8b-instruct-v1:0",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    output_text = response_body.get('generation').strip()

    print(f"Bot: {output_text}")

    history.append({
        "role": "assistant",
        "content": output_text
    })