import os
import pymupdf
import csv
import argparse
import spacy
import re
from tqdm import tqdm

class PDFParser:
    def __init__(self, pdf_dir, output_csv):
        self.pdf_dir = pdf_dir
        self.output_csv = output_csv
        self.nlp = spacy.load("en_core_web_sm")

    def is_heading(self, block_text):
        """
        Heuristic to determine if a text block is a heading.
        - Short text
        - Ends with no punctuation
        - High ratio of capital letters
        - Starts with a number
        - Starts with "Box", "Annex", or "Figure" followed by a number.
        """
        # Check for patterns like "Box 1", "Annex 1:", "Figure 1", "Figure A", "Chart 1",etc.
        if re.match(r'^(Box|Annex|Figure|Chart)\s+(\d+|[A-Z])', block_text.strip(), re.IGNORECASE):
            return True

        # Check if the text starts with a number (e.g., "1. Introduction")
        if block_text.strip() and block_text.strip()[0].isdigit():
            return True

        # Simple heuristic: short lines are often headings.
        if len(block_text.split()) < 8 and not block_text.strip().endswith(('.', ':', ';')):
            return True
        
        # More advanced checks could be added here, e.g., using font information if available
        # or NLP techniques to check for title case, etc.
        doc = self.nlp(block_text)
        # Check if the sentence is short and doesn't seem to be a full sentence.
        if len(doc) < 10 and all(token.is_title or token.is_upper for token in doc if token.is_alpha):
             return True

        return False

    def parse_pdfs(self):
        headers = ['file', 'title', 'author', 'modDate', 'creationDate', 'subject', 'textBlock']
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
            for filename in tqdm(pdf_files, desc="Parsing PDFs"):
                filepath = os.path.join(self.pdf_dir, filename)
                try:
                    doc = pymupdf.open(filepath)
                    metadata = doc.metadata
                    
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        blocks = page.get_text("blocks")
                        for block in blocks:
                            text = block[4].replace('\n', ' ').strip()
                            if text and not self.is_heading(text):
                                writer.writerow({
                                    'file': filename,
                                    'title': metadata.get('title', ''),
                                    'author': metadata.get('author', ''),
                                    'modDate': metadata.get('modDate', ''),
                                    'creationDate': metadata.get('creationDate', ''),
                                    'subject': metadata.get('subject', ''),
                                    'textBlock': text
                                })
                    doc.close()
                except Exception as e:
                    print(f"Could not process {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='''
                                     Parse PDFs from a directory, extract text blocks, and save to a CSV.
                                     ''')
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files.")
    parser.add_argument("--output_csv", type=str, default="pdf_text_blocks.csv", help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    pdf_parser = PDFParser(args.pdf_dir, args.output_csv)
    pdf_parser.parse_pdfs()
    print(f"Processing complete. Output saved to {args.output_csv}")

if __name__ == "__main__":
    main()
