import json
import os

#load file "./training_data/data_infos.json", empty it and save it
data = {"nbrOfImages": 0, "folder": "img/", "images": {}}

with open('./training_data/data_infos.json', 'w') as f:
    json.dump(data, f)

#delete all files in folder "./training_data/data"
for file in os.listdir("./training_data/img"):
    os.remove(f"./training_data/img/{file}")
