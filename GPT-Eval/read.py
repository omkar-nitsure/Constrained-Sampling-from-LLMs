import json 
import os 



with open("Files/ac2w0.00.json") as f:
    data = [json.loads(line) for line in f]

print(data[0]["generation_complete"][0:2])