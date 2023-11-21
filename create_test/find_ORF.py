import gzip
import json
import re
import sys 
def extractTranscripts(file):
   genome = dict()
   with open(file) as f:
      txt = f.read()
      for item in txt.split('>'):
         # Separate header from newline-delimited
         # sequence (proseq) and join the sequence.
         if item == "": continue
         header, proseq = item.split('\n', 1)
         seq = proseq.replace('\n', '') 
         # Keep only the first word of the header.
         genome[re.sub("\s.*", "", header)] = seq
      return genome      


genome = extractTranscripts(sys.argv[1])
longest_sequences = dict()
for identifier, sequence in genome.items():
      #parts = identifier.split("_")
      key = identifier.split('.')[0]  # Key is the first part of the identifier
      seq = sequence.split('*')
      #keep sequences with M
      seq = [s for s in sequence.split('*') if 'M' in s]
      indexes = [s.index('M') for s in seq]
      seq = [s[j:] for s, j in zip(seq, indexes)]
      #seq = [s for s in seq if s.startswith('M')]
      if key not in longest_sequences and seq != []:
         longest = max(seq, key=len)
         longest_sequences[key] = longest
      elif seq != []:
        # Compare the lengths and update if the new sequence is longer
        longest = max(seq, key=len)
        if len(longest) > len(longest_sequences[key]):
            longest_sequences[key] = longest
      else: continue

for identifier, sequence in longest_sequences.items():
   print(">"+ identifier)
   print(sequence)
