import argparse
import json
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class NERInference:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

    def load_data(self, input_file, text_column):
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported input file format. Please use CSV or Excel.")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in the input file.")
        
        return df

    def extract_entities(self, texts):
        return self.ner_pipeline(texts)

    def save_results(self, df, results, output_file, min_score):
        filtered_entities_list = []
        for entities in results:
            current_entities = []
            for entity in entities:
                if entity['score'] >= min_score:
                    # Convert numpy.float32 to float for JSON serialization
                    entity['score'] = float(entity['score'])
                    current_entities.append(entity)
            filtered_entities_list.append(current_entities)
        
        df['entities'] = filtered_entities_list
        
        if output_file.endswith('.csv'):
            df['entities'] = df['entities'].apply(json.dumps)
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            df.to_json(output_file, orient='records', lines=True, indent=4)
        else:
            print("Unsupported output file format. Saving as CSV.")
            df['entities'] = df['entities'].apply(json.dumps)
            df.to_csv(output_file, index=False)

        print(f"Inference complete. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract entities from a text file using a fine-tuned NER model.")
    parser.add_argument("input_file", help="Path to the input file (CSV or Excel).")
    parser.add_argument("output_file", help="Path to save the output file with entities (CSV or JSON).")
    parser.add_argument("--model_path", default="mdeberta-v3-econ-ie-ner-tuned", help="Path to the fine-tuned model directory.")
    parser.add_argument("--text_column", default="text", help="Name of the column containing the text to process.")
    parser.add_argument("--min_score", type=float, default=0.8, help="Minimum score for an entity to be included.")
    args = parser.parse_args()

    try:
        inference_engine = NERInference(args.model_path)
        df = inference_engine.load_data(args.input_file, args.text_column)
        texts = df[args.text_column].tolist()
        results = inference_engine.extract_entities(texts)
        inference_engine.save_results(df, results, args.output_file, args.min_score)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
