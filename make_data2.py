import json
import os

input_file_path = "create_test/train.json"  # Replace with the path to your input file
output_directory = "data2"  # Use an existing directory named "data2" for the output files

with open(input_file_path) as input_file:
   for entry in input_file:
     entry = json.loads(entry)
     entry_id = entry['id']
     if entry_id:
        file_name = f"{entry_id}.json"
        file_path = os.path.join(output_directory, file_name)

        with open(file_path, 'w') as json_file:
           json.dump(entry, json_file)





