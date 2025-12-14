"""
Aggregates OCR results from jsonl files of the form:
{"image_path": "/usr/project/xtmp/bsc32/droid/TRI/Gates/3010826835/Thu_Oct_12_12:20:30_2023/16522755/000007.png", "ocr_text": "CD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\nCD\n"}
Groups OCR results from the same scene + camera view together (eg. for scene 3010826835 and camera 16522755)
"""

import json
from collections import defaultdict
import os
import time
from tqdm import tqdm

from google import genai
from google.genai import types
from google.cloud import storage


input_files_list = [
    "/usr/project/xtmp/bsc32/droid_yoloe/AUTOLab/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/CLVR/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/GuptaLab/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/ILIAD/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/IPRL/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/IRIS/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/PennPAL/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/RAD/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/RAIL/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/REAL/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/RPL/ocr_results_0.jsonl",
    "/usr/project/xtmp/bsc32/droid_yoloe/TRI/ocr_results.jsonl",
]

output_file = "/usr/project/xtmp/bsc32/droid_ocr_aggregated_results.jsonl"

aggregated_results = {}

for input_file in input_files_list:
    with open(input_file, "r") as f:
        for line in tqdm(f, desc=f"Processing {input_file}"):
            data = json.loads(line)
            # get scene id, camera id from image path
            image_path = data["image_path"]
            path_parts = image_path.split("/")
            scene_id = path_parts[-4]
            camera_id = path_parts[-2]
            lab = path_parts[-6]
            building = path_parts[-5]

            key = f"{scene_id}_{camera_id}"
            # save file paths and ocr text to individual lists
            if key not in aggregated_results:
                aggregated_results[key] = {
                    "image_names": [],
                    "ocr_texts": [],
                    "lab": lab,
                    "building": building,
                }
            aggregated_results[key]["image_names"].append(path_parts[-1])
            aggregated_results[key]["ocr_texts"].append(data["ocr_text"])

with open(output_file, "w") as f_out:
    for key, records in aggregated_results.items():
        scene_id, camera_id = key.split("_")
        output_record = {
            "lab": records["lab"],
            "building": records["building"],
            "scene_id": scene_id,
            "camera_id": camera_id,
            "image_names": records["image_names"],
            "ocr_texts": records["ocr_texts"],
        }
        f_out.write(json.dumps(output_record) + "\n")


# # Make gemini batch job file
# # Example: {"request":{"contents": [{"role": "user", "parts": [{"text": "What is the relation between the following video and image samples?"}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/animals.mp4", "mimeType": "video/mp4"}}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/image/cricket.jpeg", "mimeType": "image/jpeg"}}]}]}}
batch_job_file = "/usr/project/xtmp/bsc32/cleanup_ocr_gemini_batch.jsonl"

with open(batch_job_file, "w") as batch_f:
    for key, records in aggregated_results.items():
        scene_id, camera_id = key.split("_")
        ocr_texts_combined = "\n".join(
            [
                f"Image: {img_name}\nOCR Text: {ocr_text}"
                for img_name, ocr_text in zip(
                    records["image_names"], records["ocr_texts"]
                )
            ]
        )
        prompt_multi = (
            f"The following are OCR results from multiple images taken in the same scene "
            f"('{scene_id}') and camera view ('{camera_id}') in the '{records['building']}' building of the '{records['lab']}' lab. "
            f"Each image may contain text recognized by an OCR system. Please help clean up and consolidate the OCR results by removing any duplicates, correcting obvious errors, "
            f"and providing a single, coherent list of unique text entries found across all images.\n\n"
            f"{ocr_texts_combined}\n\n"
            f"Respond In plain text with only the consolidated and cleaned OCR text."
        )
        prompt_single = (
            f"The following is an OCR result from an image taken in scene '{scene_id}' "
            f"and camera view '{camera_id}' in the '{records['building']}' building of the '{records['lab']}' lab. "
            f"Please help clean up the OCR result by removing any errors and providing the corrected text.\n\n"
            f"OCR Text: {records['ocr_texts'][0]}\n\n"
            f"Respond in plain text with only the cleaned OCR text."
        )
        prompt = prompt_multi if len(records["ocr_texts"]) > 1 else prompt_single
        batch_record = {
            "key": f"{scene_id}_{camera_id}",
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
        }
        batch_f.write(json.dumps(batch_record) + "\n")


client = genai.Client(
    vertexai=True, project='white-ground-476515-u3', location='us-central1'
)

# Upload file to GCP bucket
gcs_client = storage.Client()
bucket = gcs_client.bucket("genai-batch-input")
blob = bucket.blob(batch_job_file.split("/")[-1])
blob.upload_from_filename(batch_job_file)

print(f"Uploaded batch job file to gs://{bucket.name}/{blob.name}")

blob_uri = f"gs://{bucket.name}/{blob.name}"

file_batch_job = client.batches.create(
    model="gemini-2.5-flash-lite",
    src=blob_uri,
    config={
        "display_name": "cleanup_ocr",
    },
)

print(f"Created batch job: {file_batch_job.name}")

# Monitor job status
job_name = file_batch_job.name

batch_job = client.batches.get(name=job_name)

completed_states = set(
    [
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    ]
)

print(f"Polling status for job: {job_name}")
batch_job = client.batches.get(name=job_name)  # Initial get
while batch_job.state.name not in completed_states:
    print(f"Current state: {batch_job.state.name}")
    time.sleep(30)  # Wait for 30 seconds before polling again
    batch_job = client.batches.get(name=job_name)

print(f"Job finished with state: {batch_job.state.name}")
if batch_job.state.name == "JOB_STATE_FAILED":
    print(f"Error: {batch_job.error}")