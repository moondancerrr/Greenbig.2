SHELL= /bin/bash

SUBDIRS := $(wildcard ../transcriptomicData/*/)
all: $(patsubst %, %proteins.fasta, $(SUBDIRS))
%LRR.fasta: %/*.fa.gz %/*_1.txt 
	python transcribe.py $^ > $@ 

%proteins.txt: %/LRR.fasta
	transeq -sequence $^ -outseq $@ -frame 1 -table 1

%proteins.fasta: %/proteins.txt
	python find_ORF.py $^ > $@
	
%test_genes.txt: %/proteins.fasta
	makeblastdb -in $^ -dbtype prot
	blastp -query LRRproteins.fasta -db $^ -evalue 10e-4 -outfmt 10 -out $$temp.txt
	cut -f 2 -d ',' $$temp.txt | uniq > $@

all_data.json: $(SUBDIRS:%=%/*_1.json)
	python modify_dicts.py '../transcriptomicData/' $@
	sed -i "s/'/\"/g" $@

test_transcripts.json: all_data.json $(SUBDIRS:%=%/test_genes.txt)
	awk '{sub(/_1$$/, ""); print}' $(word 2, $^)  | while IFS= read -r line; do     grep "$$line" $<; done > $@
