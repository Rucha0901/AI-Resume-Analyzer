import pdfplumber

def extract_text_from_pdf(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                full_text += page_text + "\n"

    # Warn if very little text was extracted
    if len(full_text.strip()) < 100:
        print("WARNING: Very little text extracted. This PDF might be a scanned image.")

    return full_text


# Call the function
text = extract_text_from_pdf("sample_resumes/resume1.pdf")

# Print the extracted text
print(text)