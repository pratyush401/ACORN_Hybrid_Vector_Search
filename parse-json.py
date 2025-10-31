import json
import subprocess

data = []
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']:
    with open(f"listings_{i}.json", "r") as f:
        for line in f:
            data.append(json.loads(line))

print(f"Loaded {len(data)} listings")

cmd = "awk -F',' '{print $1}' map0*.csv"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

image_ids = result.stdout.strip().split('\n')

print(len(image_ids), "image IDs loaded")

i = 0
check = False
with open("metadata00.py", "w") as m_file:
    for image in image_ids:
        print(f"Processing image {i}")
        for listing in data:
            if "main_image_id" in listing:
                if listing['main_image_id'] == image:
                    check = True
                    m_file.write(f"[{image}, {json.loads(listing)}]\n")
                    break
            if "other_image_id" in listing:
                if image in listing["other_image_id"]:
                    check = True
                    m_file.write(f"[{image}, {json.loads(listing)}]\n")
                    break
        if not check:
            print("No match found for image:", image)
        check = False
            
        i = i + 1