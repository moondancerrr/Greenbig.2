import json
with open("train.json") as f:
    for line in f:
        print(json.loads(line)['id'])

