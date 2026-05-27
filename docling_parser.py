from docling.document_converter import DocumentConverter
import sys
import json

converter = DocumentConverter()

while True:
    file_path = sys.stdin.readline().strip()

    if not file_path:
        continue

    try:
        result = converter.convert(file_path)
        document = result.document
        parsed_output = []

        for item in document.texts:
            parsed_output.append({
                "text": item.text,
                "label": item.label
            })
        
        print(json.dumps(parsed_output), flush=True)
    
    except Exception as e:
        print(json.dumps({
            "error": str(e)
        }), flush=True)