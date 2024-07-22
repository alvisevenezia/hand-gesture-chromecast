import os
import json

with open('./training_data/data_infos.json') as f:
    data = json.load(f)

for file in os.listdir("./training_data/img"):
    if file.endswith(".jpg"):
                
        data['images'][data['nbrOfImages']] = ({"name": file, "label": file.replace(".jpg", ".json")})
        data['nbrOfImages'] += 1


json.dump(data, open('./training_data/data_infos.json', 'w'))
