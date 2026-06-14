from image import handler
import json

from py.src.images.titan.inpaint import response

event = {
    "body": json.dumps({"description": "A beautiful sunset"})
}

response = handler(event, {})

print(response)