import csv
import json
import argparse
import re
from ast import literal_eval
from typing import List, Dict, Any, Iterable, Tuple, Optional

class CSVTripletExtractor:
    """
    Reads a CSV with columns:
      file,title,author,modDate,creationDate,subject,textBlock,entities

    The 'entities' column contains a JSON array of dicts:
      [{"entity_group": "...", "word": "...", "score": 0.99, "start": 10, "end": 20}, ...]

    Produces a triplets CSV with columns:
      file,title,subject_text,subject_type,relation,object_text,object_type,effect_size,avg_confidence,textBlock

    Rules:
      - For each intervention I and outcome O: (I)-[impacts]->(O)
        * Attach first found effect_size (if present in the same row).
      - For each intervention I and population P: (I)-[applies_to]->(P)
      - (Optional bonus) Outcome O to Population P: (O)-[experienced_by]->(P) if enabled.
    """

    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        min_score: float = 0.0,
        attach_outcome_population: bool = False,
        keep_first_effect_only: bool = True,
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.min_score = float(min_score)
        self.attach_outcome_population = bool(attach_outcome_population)
        self.keep_first_effect_only = bool(keep_first_effect_only)

        # Valid groups we care about
        self.GROUPS = {"population", "intervention", "outcome", "effect_size"}

        # Precompiled regex to normalize percent-like effect sizes
        self.percent_like = re.compile(r"\b\d+(\.\d+)?\s?%")

    # Utilities
    def _normalize_text(self, s: str) -> str:
        if s is None:
            return ""
        # Collapse whitespace and strip quotes
        s = re.sub(r"\s+", " ", str(s)).strip().strip('“”"\'`‘’')
        return s

    def _parse_entities(self, raw: str) -> List[Dict[str, Any]]:
        """
        Parse the 'entities' field robustly:
        - Prefer json.loads
        - Fallback to ast.literal_eval
        - Return [] on failure or if not a list
        """
        if raw is None:
            return []
        txt = raw.strip()
        if not txt:
            return []
        try:
            data = json.loads(txt)
        except Exception:
            try:
                data = literal_eval(txt)
            except Exception:
                return []

        if not isinstance(data, list):
            return []
        # Keep only dicts and expected keys
        out = []
        for e in data:
            if not isinstance(e, dict):
                continue
            eg = self._normalize_text(e.get("entity_group", ""))
            w  = self._normalize_text(e.get("word", ""))
            sc = float(e.get("score", 0.0)) if e.get("score") is not None else 0.0
            if eg and w:
                out.append({"entity_group": eg.lower(), "word": w, "score": sc})
        return out

    def _filter_by_score(self, ents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.min_score <= 0.0:
            return list(ents)
        return [e for e in ents if float(e.get("score", 0.0)) >= self.min_score]

    def _collect_by_group(self, ents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        bucket: Dict[str, List[Dict[str, Any]]] = {g: [] for g in self.GROUPS}
        for e in ents:
            g = e.get("entity_group", "").lower()
            if g in bucket:
                bucket[g].append(e)
        return bucket

    def _pick_effect_sizes(self, effects: List[Dict[str, Any]]) -> Optional[str]:
        if not effects:
            return None
        # Try to find something that looks like a % first (e.g., "65% to 74%", "40% to 50%")
        # If multiple, join or take first depending on config.
        values = []
        for e in effects:
            w = e.get("word", "")
            if not w:
                continue
            # Simple normalization: extract % substrings; if none found, keep the full word.
            matches = self.percent_like.findall(w)
            if matches:
                values.append(self._normalize_text(w))
            else:
                values.append(self._normalize_text(w))
        if not values:
            return None
        if self.keep_first_effect_only:
            return values[0]
        return "; ".join(dict.fromkeys(values))  # dedup while preserving order

    def _avg_confidence(self, items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(float(i.get("score", 0.0)) for i in items) / len(items)

    #
    # Core extraction per row
    #
    def _triplets_from_row(
        self,
        row: Dict[str, Any],
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield triplet dicts with:
          subject_text, subject_type, relation, object_text, object_type, effect_size, avg_confidence
        """
        ents = self._parse_entities(row.get("entities"))
        ents = self._filter_by_score(ents)
        buckets = self._collect_by_group(ents)

        ints = buckets.get("intervention", [])
        outs = buckets.get("outcome", [])
        pops = buckets.get("population", [])
        effs = buckets.get("effect_size", [])

        effect = self._pick_effect_sizes(effs)

        # I -> O (impacts)
        for i in ints:
            for o in outs:
                yield {
                    "subject_text": i["word"],
                    "subject_type": "intervention",
                    "relation": "impacts",
                    "object_text": o["word"],
                    "object_type": "outcome",
                    "effect_size": effect,
                    "avg_confidence": round(self._avg_confidence([i, o] + (effs[:1] if effect else [])), 6),
                }

        # I -> P (applies_to)
        for i in ints:
            for p in pops:
                yield {
                    "subject_text": i["word"],
                    "subject_type": "intervention",
                    "relation": "applies_to",
                    "object_text": p["word"],
                    "object_type": "population",
                    "effect_size": None,
                    "avg_confidence": round(self._avg_confidence([i, p]), 6),
                }

        # (Optional) O -> P (experienced_by)
        if self.attach_outcome_population:
            for o in outs:
                for p in pops:
                    yield {
                        "subject_text": o["word"],
                        "subject_type": "outcome",
                        "relation": "experienced_by",
                        "object_text": p["word"],
                        "object_type": "population",
                        "effect_size": None,
                        "avg_confidence": round(self._avg_confidence([o, p]), 6),
                    }

    #
    # Pipeline
    #
    def extract(self) -> None:
        """
        Read input CSV rows, generate triplets, deduplicate, and write output CSV.
        """
        input_fields_hint = {
            'file', 'title', 'author', 'modDate', 'creationDate', 'subject', 'textBlock', 'entities'
        }
        out_fields = [
            "file", "title",
            "subject_text", "subject_type",
            "relation",
            "object_text", "object_type",
            "effect_size",
            "avg_confidence",
            "textBlock",
        ]

        dedup_key_set = set()
        to_write: List[Dict[str, Any]] = []

        with open(self.input_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            missing = input_fields_hint - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing expected input columns: {sorted(missing)}")

            for row in reader:
                base = {
                    "file": row.get("file", ""),
                    "title": row.get("title", ""),
                    "textBlock": self._normalize_text(row.get("textBlock", "")),
                }
                for t in self._triplets_from_row(row):
                    key = (
                        base["file"],
                        base["title"],
                        t["subject_text"].lower(),
                        t["subject_type"],
                        t["relation"],
                        t["object_text"].lower(),
                        t["object_type"],
                        (t["effect_size"] or "").lower(),
                    )
                    if key in dedup_key_set:
                        continue
                    dedup_key_set.add(key)

                    to_write.append({
                        **base,
                        **t
                    })

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=out_fields)
            writer.writeheader()
            for rec in to_write:
                writer.writerow(rec)


def main():
    parser = argparse.ArgumentParser(description="""
        Build PICO-style triplets from a CSV that includes an 'entities' JSON column.
        Example usage:
          python triplets.py --input_csv econ.csv --output_csv triplets.csv --min_score 0.6
    """.strip())
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV path with entity detections.")
    parser.add_argument("--output_csv", type=str, default="triplets.csv", help="Output CSV path for triplets.")
    parser.add_argument("--min_score", type=float, default=0.0, help="Minimum entity confidence to keep.")
    parser.add_argument("--attach_outcome_population", action="store_true",
                        help="Also create (outcome)-[experienced_by]->(population) edges.")
    parser.add_argument("--keep_first_effect_only", action="store_true",
                        help="If multiple effect_size mentions exist, keep only the first (default behaviour).")
    args = parser.parse_args()

    extractor = CSVTripletExtractor(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        min_score=args.min_score,
        attach_outcome_population=args.attach_outcome_population,
        keep_first_effect_only=args.keep_first_effect_only or True,
    )
    extractor.extract()
    print(f"Done. Triplets written to {args.output_csv}")

if __name__ == "__main__":
    main()