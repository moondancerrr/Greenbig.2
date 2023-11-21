#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gzip
import json
import re
import sys

'''
python fasta_extractor.py <fname.fasta> <start> <length>
'''

def rc(seq):
   replace = { "A":"T", "C":"G", "G":"C", "T":"A", "N":"N",
               "a":"t", "c":"g", "g":"c", "t":"a", "n":"n" }
   return "".join(replace.get(x, "N") for x in seq[::-1])

def dict_from_fasta_file(f):
   '''
   Read the whole file in memory. This can crash if the file is big.
   '''
   genome = dict()
   txt = f.read() # Read it all.
   # Split on fasta header
   for item in txt.decode("ascii").split('>'):
      # Separate header from newline-delimited
      # sequence (proseq) and join the sequence.
      if item == "": continue
      header, proseq = item.split('\n', 1)
      seq = proseq.replace('\n', '')
      # Keep only the first word of the header.
      genome[re.sub("\s.*", "", header)] = seq
   return genome

if __name__ == '__main__':
   fastagz  = sys.argv[1]
   coordgz = sys.argv[2]
   with gzip.open(fastagz) as f:
      genome = dict_from_fasta_file(f)
   with open(coordgz) as f:
      for line in f:
         #items = line.decode("ascii").split()
         items = line.split()
         #exon = items[0]
         name = items[0]
         chrom = items[1]
         start = int(items[2])
         end = int(items[3])
         strand = items[4]
         exon_p = items[5]
         exon_l = items[6]
         
         if chrom not in genome:
            continue
         seq = genome[chrom][(start-1):end]
         #calls = list()
         transcribed_seq = ""
         for l_, p_ in zip(exon_l.split(","), exon_p.split(",")):
            if p_ == "" or l_ == "": continue
            pos, length = int(p_), int(l_)
            #calls += [0] * (l-len(calls)) + [1] * p
            transcribed_seq += seq[pos:pos+length]
         print(">" + name)
         if strand == "+":
            print(seq)
         else:
            print(rc(seq))
         """
         if strand == "+":
            print(json.dumps({"id": name, "seq": seq, "calls": calls}))
         else:
            print(json.dumps({"id": name, "seq": rc(seq), "calls": calls[::-1]}))
         """
