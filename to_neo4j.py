#!/usr/bin/env python3
import os
import argparse

class CypherFromTripletsWithFTS:
    """
    Generate a Cypher script from a triplets CSV that:
      - Creates uniqueness constraints
      - Creates full-text indexes (Concept, Excerpt, Document) for Neo4j 5
      - Loads CSV and builds the graph (no APOC required)

    Expected CSV columns:
      file,title,subject_text,subject_type,relation,object_text,object_type,effect_size,avg_confidence,textBlock
    """

    def __init__(self, input_csv: str, output_cypher: str):
        self.input_csv = input_csv
        self.output_cypher = output_cypher

    def _script(self, csv_basename: str) -> str:
        # Note: Double braces {{ }} are used to emit single { } into the Cypher output from an f-string.
        return f"""// Auto-generated Cypher import script with Full-Text Indexes (Neo4j 5)
// Place {csv_basename} into Neo4j's import/ directory.

/// ---- Constraints ----
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept)  REQUIRE c.unique_key IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_key    IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Excerpt)  REQUIRE e.excerpt_key IS UNIQUE;

/// ---- Full-text indexes (Neo4j 5 syntax) ----
CREATE FULLTEXT INDEX concept_text_fts IF NOT EXISTS
FOR (c:Concept) ON EACH [c.text]
OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'english', `fulltext.eventually_consistent`: true }} }};

CREATE FULLTEXT INDEX excerpt_text_fts IF NOT EXISTS
FOR (e:Excerpt) ON EACH [e.text, e.title]
OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'english', `fulltext.eventually_consistent`: true }} }};

CREATE FULLTEXT INDEX document_title_fts IF NOT EXISTS
FOR (d:Document) ON EACH [d.title]
OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'english', `fulltext.eventually_consistent`: true }} }};

/// ---- Param for the CSV file name ----
:param csvFile => '{csv_basename}';

/// ---- Import pipeline ----
LOAD CSV WITH HEADERS FROM 'file:///' + $csvFile AS row
WITH row
WHERE trim(coalesce(row.subject_text,'')) <> '' AND trim(coalesce(row.object_text,'')) <> ''

WITH
  row,
  toLower(trim(row.subject_type)) AS s_type,
  toLower(trim(row.object_type))  AS o_type,
  toLower(trim(row.relation))     AS rel_lc,
  trim(row.subject_text)          AS s_text,
  trim(row.object_text)           AS o_text,
  trim(coalesce(row.file,''))     AS file,
  trim(coalesce(row.title,''))    AS title,
  trim(coalesce(row.textBlock,'')) AS textBlock,
  trim(coalesce(row.effect_size,'')) AS effect_size,
  toFloat(coalesce(row.avg_confidence,'0')) AS avg_confidence

WITH
  row, s_type, o_type, rel_lc, s_text, o_text, file, title, textBlock, effect_size, avg_confidence,
  (s_type + '|' + toLower(s_text)) AS s_key,
  (o_type + '|' + toLower(o_text)) AS o_key,
  (file + '|' + title) AS doc_key,
  (file + '|' + title + '|' + left(textBlock, 1024)) AS excerpt_key

// Document & Excerpt (provenance)
MERGE (d:Document {{doc_key: doc_key}})
  ON CREATE SET d.file = file, d.title = title

MERGE (x:Excerpt {{excerpt_key: excerpt_key}})
  ON CREATE SET
    x.file  = file,
    x.title = CASE
                WHEN textBlock IS NULL THEN ''
                WHEN size(textBlock) > 120 THEN substring(textBlock, 0, 120) + '…'
                ELSE textBlock
              END,
    x.text  = textBlock


MERGE (d)-[:HAS_EXCERPT]->(x)

// Subject Concept
MERGE (s:Concept {{unique_key: s_key}})
  ON CREATE SET s.text = s_text, s.type = s_type
FOREACH (_ IN CASE WHEN s_type = 'intervention' THEN [1] ELSE [] END | SET s:Intervention)
FOREACH (_ IN CASE WHEN s_type = 'outcome'      THEN [1] ELSE [] END | SET s:Outcome)
FOREACH (_ IN CASE WHEN s_type = 'population'   THEN [1] ELSE [] END | SET s:Population)

// Object Concept
MERGE (o:Concept {{unique_key: o_key}})
  ON CREATE SET o.text = o_text, o.type = o_type
FOREACH (_ IN CASE WHEN o_type = 'intervention' THEN [1] ELSE [] END | SET o:Intervention)
FOREACH (_ IN CASE WHEN o_type = 'outcome'      THEN [1] ELSE [] END | SET o:Outcome)
FOREACH (_ IN CASE WHEN o_type = 'population'   THEN [1] ELSE [] END | SET o:Population)

// Provenance mentions
MERGE (s)-[:MENTIONED_IN]->(x)
MERGE (o)-[:MENTIONED_IN]->(x)

// Relationships without APOC (FOREACH/CASE to choose rel type)
FOREACH (_ IN CASE WHEN rel_lc = 'impacts' THEN [1] ELSE [] END |
  MERGE (s)-[r:IMPACTS]->(o)
    ON CREATE SET r.effect_size = CASE WHEN effect_size = '' THEN NULL ELSE effect_size END
  SET r.avg_confidence = avg_confidence, r.file = file, r.title = title
)
FOREACH (_ IN CASE WHEN rel_lc = 'applies_to' THEN [1] ELSE [] END |
  MERGE (s)-[r:APPLIES_TO]->(o)
    ON CREATE SET r.effect_size = CASE WHEN effect_size = '' THEN NULL ELSE effect_size END
  SET r.avg_confidence = avg_confidence, r.file = file, r.title = title
)
FOREACH (_ IN CASE WHEN rel_lc = 'experienced_by' THEN [1] ELSE [] END |
  MERGE (s)-[r:EXPERIENCED_BY]->(o)
    ON CREATE SET r.effect_size = CASE WHEN effect_size = '' THEN NULL ELSE effect_size END
  SET r.avg_confidence = avg_confidence, r.file = file, r.title = title
);
"""

    def generate(self) -> None:
        csv_basename = os.path.basename(self.input_csv)
        with open(self.output_cypher, "w", encoding="utf-8") as f:
            f.write(self._script(csv_basename))

def main():
    parser = argparse.ArgumentParser(description="""
        Generate a Cypher import script (with Neo4j 5 full-text indexes) from a triplets CSV.
        Usage:
          python cypher_from_triplets_fts.py --input_csv triplets.csv --output_cypher import_triplets.cypher
    """.strip())
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the triplets CSV.")
    parser.add_argument("--output_cypher", type=str, default="import_triplets.cypher", help="Output Cypher script path.")
    args = parser.parse_args()

    gen = CypherFromTripletsWithFTS(args.input_csv, args.output_cypher)
    gen.generate()
    print(f"Wrote {args.output_cypher}. Copy {os.path.basename(args.input_csv)} to Neo4j import/ and run the script.")

if __name__ == "__main__":
    main()
