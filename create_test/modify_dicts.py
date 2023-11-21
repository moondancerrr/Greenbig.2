import os
import json
import sys
import numpy as np

base_dir = sys.argv[1]  # Replace with the actual base directory path
with open(sys.argv[2], "w") as output:
   for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        for file_name in os.listdir(subdir_path):
           if file_name.endswith("_1.json") and not file_name.startswith("update"):
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path) as json_file:
                #with open(file_path) as json_file:
                   for line in json_file:
                     data = json.loads(line)
                     # Add the 'species' field
                     data["species"] = subdir
                     if "field" in data:
                        del data["field"]
                     # Save the modified JSON file
                     output.write(json.dumps(data) + "\n")
