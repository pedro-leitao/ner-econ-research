# Named Entity Recognition for Economic Research and Knowledge Graphs

Economic Research is a field that relies heavily on the analysis of large volumes of text data, such as news articles, financial reports, and social media posts. Named Entity Recognition (NER) is a crucial task in natural language processing (NLP) that involves identifying and classifying named entities in text into predefined categories such as persons, organizations, locations, dates, and monetary values. Beyond these standard categories, NER can also be used for more specialized entities relevant to economic research, such as capturing *efects in policy interventions* - this is particularly useful for government agencies, policy makers, and researchers who need to analyze the impact of various policies on economic outcomes.

This repository provides a collection of tools for performing NER on economic text data, with a focus on identifying *causal knowledge extraction* and *evidence based policy making*.

## The tooling

### PDF text extraction

The `pdf_parser.py` script extracts text blocks from PDF files in a specified directory. It uses heuristics to distinguish between content and headings, aiming to extract meaningful paragraphs while discarding titles and section numbers. The extracted text, along with metadata like file name and title, is saved to a CSV file for further processing.

### Fine-tuning encoder models for NER

`trainer.py` is used to fine-tune a Hugging Face transformer model for Named Entity Recognition (NER) on a custom dataset. It takes a CoNLL-formatted text file as input, where tokens and their corresponding NER tags are defined. The script handles data loading, tokenization, alignment of labels, and the training process itself. After training, it saves the fine-tuned model and provides an evaluation on a test set.

For the purpose of causal knowledge extraction, this repository includes the ECON-IE NER training dataset in the `data/econ_ie` directory. You can use it to fine-tune models for this specific task.

For example, to fine-tune the `worldbank/econberta-fs` model, you can run the following command:

```bash
python trainer.py \
  --model_name worldbank/econberta-fs \
  --train_file data/econ_ie/train.conll \
  --validation_file data/econ_ie/dev.conll \
  --test_file data/econ_ie/test.conll \
  --output_dir econberta-fs-econ-ie-ner-tuned \
  --epochs 5
```

### Extracting entities

With a fine-tuned model available, `inference-ner.py` script is used to perform NER on a larger set of texts. It reads text from a specified column in a CSV file, runs the NER pipeline on each text, and appends the extracted entities as a new column in the output file. It allows setting a minimum confidence score to filter out less certain entities.

### Knowledge graph construction

This part of the toolchain converts the extracted entities into a knowledge graph. It's a two-step process:
1.  **`relate.py`**: This script reads the CSV file containing the entities and applies a set of rules to create relationships (triplets) between them. For example, it can link an `intervention` entity to an `outcome` entity with an "impacts" relationship. The resulting triplets are saved to a new CSV file.
2.  **`to_neo4j.py`**: This script takes the triplets CSV and generates a Cypher script. This script can be run in a Neo4j database to create the knowledge graph, including nodes for concepts and documents, and the relationships between them. It also sets up constraints and full-text search indexes for efficient querying.