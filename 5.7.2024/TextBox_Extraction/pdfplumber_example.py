import pdfplumber

def extract_text_content(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page=pdf.pages[1]
        text=page.extract_text()
        return text

content = extract_text_content(".\PDF/10.[SPQ] Seller Property Questionnaire.pdf")
# content = extract_text_content('.\PDF\8. [TDS] Real Estate Transfer Disclosure Statement.pdf')

lines = content.split('\n')


i = 0
while i < len(lines):
    if lines[i] == 'n':
        del lines[i:i+2] 
    elif lines[i] == 'nX':
        del lines[i]
    else:
        i += 1

content_with_checked_content = '\n'.join(lines)

print(content)
