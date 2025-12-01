import os
import json
import pprint
import sys

if len(sys.argv) != 2:
     print("Error: need an argument")
     sys.exit()

uniqueness = {}
with open("metadata-small.py", "r") as meta_small:
    for line in meta_small:
            curr_line = line[1:-1]
            curr_line = curr_line.strip().split(",")
            image_id = curr_line[0]
            metadata = ",".join(curr_line[1:])[:-1]
            temp_dict = json.loads(metadata)
            if sys.argv[1] == "country":
                if sys.argv[1] in temp_dict.keys():
                    if temp_dict[sys.argv[1]] in uniqueness.keys():
                        uniqueness[temp_dict[sys.argv[1]]] = uniqueness[temp_dict[sys.argv[1]]] + 1
                    else:
                        uniqueness[temp_dict[sys.argv[1]]] = 0
            elif sys.argv[1] == "item_weight":
                if sys.argv[1]["normalized_value"]["value"] in temp_dict.keys():
                    if temp_dict[sys.argv[1]]["normalized_value"]["value"] in uniqueness.keys():
                        uniqueness[temp_dict[sys.argv[1]]["normalized_value"]["value"]] = uniqueness[temp_dict[sys.argv[1]]["normalized_value"]["value"]] + 1
                    else:
                        uniqueness[temp_dict[sys.argv[1]]["normalized_value"]["value"]] = 0 
            else:
                if sys.argv[1] in temp_dict.keys():
                    if temp_dict[sys.argv[1]][0]["value"] in uniqueness.keys():
                        uniqueness[temp_dict[sys.argv[1]][0]["value"]] = uniqueness[temp_dict[sys.argv[1]][0]["value"]] + 1
                    else:
                        uniqueness[temp_dict[sys.argv[1]][0]["value"]] = 0

pprint.pprint(uniqueness)