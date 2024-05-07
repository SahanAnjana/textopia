import fitz  # PyMuPDF

def extract_filled_content(pdf_path):
    filled_content = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        annotations = page.annots()
        for annot in annotations:
            if annot.type[1] == 4:  # Check if the annotation is a text annotation
                filled_content.append(annot.info["content"])  # Extract the filled content
    doc.close()
    return filled_content

# Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_path = 'pdf1.pdf'
filled_content = extract_filled_content(pdf_path)
for content in filled_content:
    print(content)
