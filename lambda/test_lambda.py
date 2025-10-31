import json
from app import handler

with open("lambda/event.json") as f:
    event = json.load(f)

result = handler(event, None)
print(result)
