import json
import subprocess

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
# The mapping files (map0*.csv) contain image ID information.
# The AWK command prints the first column of each CSV row.
# --------------------------------------------------------------
cmd = "awk -F',' '{print $1}' map0*.csv"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

image_ids = result.stdout.strip().split('\n') # Parse the extracted image IDs (one per line)

print(len(image_ids), "image IDs loaded")

# --------------------------------------------------------------
# Loop through all image IDs and attempt to find matching
# listings that reference them via:
#  - main_image_id
#  - other_image_id (a list)
#
# For each match, write a simplified record into metadata00.py
# in the form: [image_id, listing_json]
# --------------------------------------------------------------
i = 0
check = False
with open("metadata00.py", "w") as m_file:
    # Iterate through each image ID that needs metadata
    for image in image_ids:
        print(f"Processing image {i}")
        # Search through all listings for a match
        for listing in data:
            # Direct match with main image field
            if "main_image_id" in listing:
                if listing['main_image_id'] == image:
                    check = True
                    # Write found metadata to output file
                    m_file.write(f"[{image}, {json.loads(listing)}]\n")
                    break
            # Match against other images
            if "other_image_id" in listing:
                if image in listing["other_image_id"]:
                    check = True
                    m_file.write(f"[{image}, {json.loads(listing)}]\n")
                    break
        # Report missing matches for debugging/tracking
        if not check:
            print("No match found for image:", image)
        # Reset match tracking flag for next image ID
        check = False   
        i = i + 1
