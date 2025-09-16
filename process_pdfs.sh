#!/bin/sh
#
# Assumes you have created the conda environment `econ-ie` with:
#
# conda env create -f environment.yml
#
# And activated it with:
#
# conda activate econ-ie
#
python -m spacy download en_core_web_sm
python pdf_parser.py --pdf_dir pdfs --output_csv results/all_pdfs.csv --min_chars 60 --max_sents 1
python inference-ner.py results/all_pdfs.csv results/all_pdfs_inferred.csv --model_path models/econberta-fs-econ-ie-ner-tuned --text_column textBlock --min_score 0.9
python relate.py --input_csv results/all_pdfs_inferred.csv --output_csv results/all_pdfs_related.csv --attach_outcome_population
python to_neo4j.py --input_csv all_pdfs_related.csv --output_cypher results/all_pdfs_related.cypher
