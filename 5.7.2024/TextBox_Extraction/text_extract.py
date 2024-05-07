from PyPDF2 import PdfReader

pdf=open("pdf1.pdf","rb")

reader=PdfReader(pdf)

page=reader.pages[1]
# fields=reader.get_fields()

# field_values={}
# for field_name,field_data in fields.items():
#     field_values[field_name]=field_data.get('/V',None)
#     print(f"Name: {field_name}\t Value: {field_data}\n")
    

for annot in page['/Annots']:
    annotation = annot.get_object()
    print(annotation)


# import fitz
# import pandas as pd 
# doc = fitz.open('pdf1.pdf')
# page1 = doc[0]
# words = page1.get_text("words")