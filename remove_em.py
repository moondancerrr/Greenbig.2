import json
import sys
import gzip
with gzip.open(sys.argv[1], 'rt') as f:
   for line in f:
      doc = json.loads(line)
      if len(doc["seq"]) > 800:
         print(json.dumps(doc))
