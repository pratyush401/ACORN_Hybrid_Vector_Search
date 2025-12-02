import json
import subprocess
import os

# --------------------------------------------------------------
# Load all listing JSON lines from multiple listing files
# The ABO dataset is partitioned across multiple JSONL files
# named listings_0.json, listings_1.json, ..., listings_f.json.
# Each line in these files represents a single product listing.
# --------------------------------------------------------------
data = []
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']:
    # Open each JSON lines file and load each line as a JSON object
    with open(f"listings_{i}.json", "r") as f:
        for line in f:
            data.append(json.loads(line))

print(f"Loaded {len(data)} listings")

# --------------------------------------------------------------
# Extract image IDs from CSV mapping files using AWK.
# The mapping files (map*.csv) contain image ID information.
# The AWK command prints the first column of each CSV row.
# --------------------------------------------------------------
cmd = "awk -F',' '{print $1}' map*.csv"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

image_ids = result.stdout.strip().split('\n') # Parse the extracted image IDs (one per line)

print(len(image_ids), "image IDs loaded")

# --------------------------------------------------------------
# Loop through all image IDs and attempt to find matching
# listings that reference them via:
#  - main_image_id
#  - other_image_id (a list)
#
# For each match, write a simplified record into metadata.py
# in the form: [image_id, metadata_json]
# --------------------------------------------------------------
i = 0
check = False
if not os.path.exists("metadata-small.py"):
    with open("metadata.py", "w") as m_file:
        for image in image_ids:
            print(f"image {i}", end=", ")
            for listing in data:
                if "main_image_id" in listing:
                    if listing['main_image_id'] == image:
                        check = True
                        m_file.write(f"[{image}, {json.dumps(listing)}]\n")
                        break
                if "other_image_id" in listing:
                    if image in listing["other_image_id"]:
                        check = True
                        m_file.write(f"[{image}, {json.dumps(listing)}]\n")
                        break
            if not check:
                print("No match found for image:", image)
            check = False
            i = i + 1

# --------------------------------------------------------------
# Select the required metadata and output a file that contains 
# only the required attributes in the json objects
# --------------------------------------------------------------

if not os.path.exists("metadata-small.py"):
    temp_dict = {}
    with open("metadata.py", "r") as f:
        with open("metadata-small.py", "w") as f_write:
            for line in f:
                curr_line = line[1:-1]
                curr_line = curr_line.strip().split(",")
                image_id = curr_line[0]
                metadata = ",".join(curr_line[1:])[:-1]
                for key in json.loads(metadata).keys():
                    if key in ["color", "item_weight", "model_year", "brand", "country"]:
                        temp_dict[key] =  json.loads(metadata)[key]
                f_write.write(f"[{image_id}, {json.dumps(temp_dict)}]\n")
                temp_dict = {}
