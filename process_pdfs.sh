#!/bin/sh
#
# Assumes you have created the conda environment `econ-ie` with:
#
# conda env create -f environment.yml
#
conda activate econ-ie
python -m spacy download en_core_web_sm
#
# Pick a batch size that fits your GPU memory, an NVidia RTX 3900 with 24GB of RAM will fit a --batch_size 32
#
python trainer.py --model_name worldbank/econberta-fs --train_file data/econ_ie/train.conll --validation_file data/econ_ie/dev.conll --test_file data/econ_ie/test.conll --output_dir models/econberta-fs-econ-ie-ner-tuned --epochs 5 --batch 32
python pdf_parser.py --pdf_dir pdfs --output_csv results/all_pdfs.csv
python inference-ner.py results/all_pdfs.csv results/all_pdfs_inferred.csv --model_path models/econberta-fs-econ-ie-ner-tuned --text_column textBlock
python relate.py --input_csv results/all_pdfs_inferred.csv --output_csv results/all_pdfs_related.csv --attach_outcome_population
python to_neo4j.py --input_csv all_pdfs_related.csv --output_cypher results/all_pdfs_related.cypher
