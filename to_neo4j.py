import os
import argparse

class CypherFromTripletsWithFTS:
    """
    Generate a Cypher script from a triplets CSV that:
      - Creates uniqueness constraints per concrete label
      - Creates full-text indexes (Intervention, Outcome, Population, Coreference, Excerpt, Document) for Neo4j 5
      - Loads CSV and builds the graph (no APOC required)

    Expected CSV columns:
      file,title,subject_text,subject_type,relation,object_text,object_type,effect_size,avg_confidence,textBlock

    Allowed *_type values (case-insensitive): intervention, outcome, population, coreference
    """

    def __init__(self, input_csv: str, output_cypher: str):
        self.input_csv = input_csv
        self.output_cypher = output_cypher

    def _script(self, csv_basename: str) -> str:
        # Note: Double braces {{ }} emit single { } inside f-strings.
        return f"""// Auto-generated Cypher import script (concrete labels, Neo4j 5)
// Place {csv_basename} into Neo4j's import/ directory.

/// ---- Constraints (per concrete label) ----
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Intervention) REQUIRE n.unique_key IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Outcome)      REQUIRE n.unique_key IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Population)   REQUIRE n.unique_key IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Coreference)  REQUIRE n.unique_key IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_key    IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Excerpt)  REQUIRE e.excerpt_key IS UNIQUE;

/// ---- Full-text indexes (Neo4j 5 syntax) ----
CREATE FULLTEXT INDEX intervention_text_fts IF NOT EXISTS
FOR (n:Intervention) ON EACH [n.text]
OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'english', `fulltext.eventually_consistent`: true }} }};

CREATE FULLTEXT INDEX outcome_text_fts IF NOT EXISTS
FOR (n:Outcome) ON EACH [n.text]
OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'english', `fulltext.eventually_consistent`: true }} }};

CREATE FULLTEXT INDEX population_text_fts IF NOT EXISTS
FOR (n:Population) ON EACH [n.text]
OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'english', `fulltext.eventually_consistent`: true }} }};

CREATE FULLTEXT INDEX coref_text_fts IF NOT EXISTS
FOR (n:Coreference) ON EACH [n.text]
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
  ON CREATE SET d.file = file, d.title = title;

MERGE (x:Excerpt {{excerpt_key: excerpt_key}})
  ON CREATE SET
    x.file  = file,
    x.title = CASE
                WHEN textBlock IS NULL THEN ''
                WHEN size(textBlock) > 120 THEN substring(textBlock, 0, 120) + '…'
                ELSE textBlock
              END,
    x.text  = textBlock;

MERGE (d)-[:HAS_EXCERPT]->(x);

// --- Subject node as concrete label (no APOC) ---
CALL {{
  WITH s_type, s_key, s_text
  CALL {{
    WITH s_type, s_key, s_text
    WHERE s_type = 'intervention'
    MERGE (n:Intervention {{unique_key: s_key}})
      ON CREATE SET n.text = s_text
    RETURN n
    UNION
    WITH s_type, s_key, s_text
    WHERE s_type = 'outcome'
    MERGE (n:Outcome {{unique_key: s_key}})
      ON CREATE SET n.text = s_text
    RETURN n
    UNION
    WITH s_type, s_key, s_text
    WHERE s_type = 'population'
    MERGE (n:Population {{unique_key: s_key}})
      ON CREATE SET n.text = s_text
    RETURN n
    UNION
    WITH s_type, s_key, s_text
    WHERE s_type = 'coreference'
    MERGE (n:Coreference {{unique_key: s_key}})
      ON CREATE SET n.text = s_text
    RETURN n
  }}
  RETURN n AS s
}}

// --- Object node as concrete label (no APOC) ---
CALL {{
  WITH o_type, o_key, o_text
  CALL {{
    WITH o_type, o_key, o_text
    WHERE o_type = 'intervention'
    MERGE (n:Intervention {{unique_key: o_key}})
      ON CREATE SET n.text = o_text
    RETURN n
    UNION
    WITH o_type, o_key, o_text
    WHERE o_type = 'outcome'
    MERGE (n:Outcome {{unique_key: o_key}})
      ON CREATE SET n.text = o_text
    RETURN n
    UNION
    WITH o_type, o_key, o_text
    WHERE o_type = 'population'
    MERGE (n:Population {{unique_key: o_key}})
      ON CREATE SET n.text = o_text
    RETURN n
    UNION
    WITH o_type, o_key, o_text
    WHERE o_type = 'coreference'
    MERGE (n:Coreference {{unique_key: o_key}})
      ON CREATE SET n.text = o_text
    RETURN n
  }}
  RETURN n AS o
}}

// Provenance mentions
MERGE (s)-[:MENTIONED_IN]->(x);
MERGE (o)-[:MENTIONED_IN]->(x);

// Relationships without APOC (choose rel type by value)
FOREACH (_ IN CASE WHEN rel_lc = 'impacts' THEN [1] ELSE [] END |
  MERGE (s)-[r:IMPACTS]->(o)
    ON CREATE SET r.effect_size = CASE WHEN effect_size = '' THEN NULL ELSE effect_size END
  SET r.avg_confidence = avg_confidence, r.file = file, r.title = title
);
FOREACH (_ IN CASE WHEN rel_lc = 'applies_to' THEN [1] ELSE [] END |
  MERGE (s)-[r:APPLIES_TO]->(o)
    ON CREATE SET r.effect_size = CASE WHEN effect_size = '' THEN NULL ELSE effect_size END
  SET r.avg_confidence = avg_confidence, r.file = file, r.title = title
);
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
        Generate a Cypher import script (Neo4j 5) from a triplets CSV using concrete labels (Intervention/Outcome/Population/Coreference).
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