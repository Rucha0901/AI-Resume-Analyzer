# test_extractor.py

from pdf_extractor import extract_text_from_pdf, clean_text, remove_stopwords
from entity_extractor import extract_all_entities, print_results

def analyze_resume(pdf_path):
    print(f"\nAnalyzing: {pdf_path}")
    
    # Step 1: Get text from PDF (from your Day 3-4 code)
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Clean it
    cleaned_text = clean_text(raw_text)
    
    # Step 3: Extract entities
    results = extract_all_entities(raw_text, cleaned_text)
    
    # Step 4: Print results
    print_results(results)
    
    return results


# Test on all your sample resumes
resume_files = [
    "sample_resumes/resume1.pdf",
    "sample_resumes/resume2.pdf",
    "sample_resumes/resume3.pdf",
]

for resume in resume_files:
    analyze_resume(resume)
    input("\nPress Enter to see next resume...")  # pause between resumes