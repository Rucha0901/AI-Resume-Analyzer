import pdfplumber

# Open the PDF file
with pdfplumber.open("sample_resumes/resume1.pdf") as pdf:
    
    # A PDF can have multiple pages - loop through all of them
    for page_number, page in enumerate(pdf.pages):
        
        # Extract the text from this one page
        text = page.extract_text()
        
        print(f"--- PAGE {page_number + 1} ---")
        print(text)
        print()  # blank line between pages
        