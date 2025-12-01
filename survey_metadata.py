import os
import json
import pprint
import sys

# --------------------------------------------------------------
# This script inspects metadata-small.py and computes the
# distribution of unique values for a single metadata key.
#
# Usage example:
#     python survey_metadata.py country
#
# It prints how many times each unique metadata value appears.
#
# NOTE: This script assumes the metadata-small.py file contains
#       lines in the form: [image_id, {metadata_dict}]
# --------------------------------------------------------------

# Ensure exactly one argument is passed (the metadata key to inspect)
if len(sys.argv) != 2:
     print("Error: need an argument")
     sys.exit()

# Dictionary for counting occurrences of each unique metadata value
uniqueness = {}
# --------------------------------------------------------------
# Open the metadata-small.py file and scan each line for the
# desired metadata attribute.
# --------------------------------------------------------------
with open("metadata-small.py", "r") as meta_small:
    for line in meta_small:
            # Each line looks like: [image_id, {...json...}]
            # Remove surrounding brackets
            curr_line = line[1:-1]
            # Split into image_id and the JSON metadata segment
            curr_line = curr_line.strip().split(",")
            image_id = curr_line[0]
            # Re-join remaining tokens into a valid JSON string
            metadata = ",".join(curr_line[1:])[:-1]
            temp_dict = json.loads(metadata) # Parse metadata JSON into a dictionary
            # --------------------------------------------------
            # Case 1: If querying "country" (this one appears
            #         to be stored as a plain string)
            # --------------------------------------------------
            if sys.argv[1] == "country":
                if sys.argv[1] in temp_dict.keys():
                    # Increment or initialize country count
                    if temp_dict[sys.argv[1]] in uniqueness.keys():
                        uniqueness[temp_dict[sys.argv[1]]] = uniqueness[temp_dict[sys.argv[1]]] + 1
                    else:
                        uniqueness[temp_dict[sys.argv[1]]] = 0
             # --------------------------------------------------
            # Case 2: If querying "item_weight"
            #         Format: temp_dict["item_weight"][0]["normalized_value"]["value"]
            #
            # NOTE: The code checks sys.argv[1]["normalized_value"]["value"],
            #         which is likely not correct but left unchanged as requested.
            # --------------------------------------------------
            elif sys.argv[1] == "item_weight":
                if sys.argv[1]["normalized_value"]["value"] in temp_dict.keys():
                    # Extract weight value and update count
                    if temp_dict[sys.argv[1]]["normalized_value"]["value"] in uniqueness.keys():
                        uniqueness[temp_dict[sys.argv[1]]["normalized_value"]["value"]] = uniqueness[temp_dict[sys.argv[1]]["normalized_value"]["value"]] + 1
                    else:
                        uniqueness[temp_dict[sys.argv[1]]["normalized_value"]["value"]] = 0 
            # --------------------------------------------------
            # Case 3: All other metadata keys
            #         Format: temp_dict[key][0]["value"]
            # --------------------------------------------------
            else:
                if sys.argv[1] in temp_dict.keys():
                    # Extract the nested metadata value and update frequency
                    if temp_dict[sys.argv[1]][0]["value"] in uniqueness.keys():
                        uniqueness[temp_dict[sys.argv[1]][0]["value"]] = uniqueness[temp_dict[sys.argv[1]][0]["value"]] + 1
                    else:
                        uniqueness[temp_dict[sys.argv[1]][0]["value"]] = 0
# --------------------------------------------------------------
# Print the dictionary showing counts for each unique metadata
# value related to the requested key.
# --------------------------------------------------------------
pprint.pprint(uniqueness)
