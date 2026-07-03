import pdfplumber
import re          # "re" stands for Regular Expressions - a tool for finding/replacing text patterns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if you haven't already (safe to run multiple times)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ============================================================
# FUNCTION 1: Extract raw text from a PDF file
# ============================================================
def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts all text from all pages.
    Returns one big string with all the text combined.
    
    pdf_path = the file path to your PDF, e.g. "sample_resumes/resume1.pdf"
    """
    
    full_text = ""  # We'll build up the text here, page by page
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            
            for page in pdf.pages:
                # extract_text() reads all the text from one page
                page_text = page.extract_text()
                
                # Some pages might be empty (like a blank page) - skip those
                if page_text:
                    full_text += page_text + "\n"  # Add each page's text + a newline separator
        
        return full_text
    
    except Exception as e:
        # If the file doesn't exist, or is corrupted, tell us what went wrong
        print(f"Error reading PDF: {e}")
        return ""


# ============================================================
# FUNCTION 2: Clean the raw text
# ============================================================
def clean_text(raw_text):
    """
    Takes messy raw text and cleans it up step by step.
    Returns clean text ready for processing.
    """
    
    # --- Step A: Fix encoding issues ---
    # Sometimes PDFs have weird characters like \x00 (null bytes) or 
    # special unicode symbols. We encode to ASCII and ignore unknown characters.
    text = raw_text.encode('ascii', errors='ignore').decode('ascii')
    
    # --- Step B: Replace special bullet points and dashes ---
    # Bullet points (•, ▪, ◦) become empty string (just remove them)
    # Em dashes (–, —) become regular hyphens
    text = text.replace('•', ' ')
    text = text.replace('▪', ' ')
    text = text.replace('◦', ' ')
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    
    # --- Step C: Remove URLs (website links) ---
    # Pattern explanation: http or https, then ://, then any non-space characters
    # re.sub means "find this pattern and replace it with something"
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # --- Step D: Remove email addresses ---
    # We'll extract emails separately later, so remove them from the main text
    # Pattern: characters@characters.characters
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # --- Step E: Remove phone numbers ---
    # Pattern: optional +, then digits, spaces, hyphens, brackets
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' ', text)
    
    # --- Step F: Remove special characters, keep only letters, numbers, spaces ---
    # [^a-zA-Z0-9\s] means "anything that is NOT a letter, number, or space"
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # --- Step G: Remove extra whitespace ---
    # \s+ means "one or more whitespace characters" (spaces, tabs, newlines)
    # Replace all of them with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # --- Step H: Strip leading and trailing spaces ---
    text = text.strip()
    
    # --- Step I: Convert to lowercase ---
    text = text.lower()
    
    return text


# ============================================================
# FUNCTION 3: Remove stopwords from cleaned text
# ============================================================
def remove_stopwords(clean_text_input):
    """
    Splits text into words, removes common useless words (stopwords),
    and returns the filtered text as a string.
    """
    
    # Load English stopwords list from NLTK
    stop_words = set(stopwords.words('english'))
    
    # Split text into individual words (tokenize)
    tokens = word_tokenize(clean_text_input)
    
    # Keep only words that:
    # 1. Are NOT stopwords
    # 2. Are alphabetic (actual words, not numbers or leftover symbols)
    # 3. Are longer than 1 character (removes lone 'a', 'i', etc.)
    filtered_tokens = [
        word for word in tokens 
        if word not in stop_words 
        and word.isalpha() 
        and len(word) > 1
    ]
    
    # Join the words back into a single string
    return " ".join(filtered_tokens)


# ============================================================
# FUNCTION 4: The master function that does everything
# ============================================================
def process_resume(pdf_path):
    """
    Takes a PDF file path, runs all steps, and returns:
    - raw_text: what came directly out of the PDF
    - cleaned_text: after cleaning symbols, spaces etc.
    - final_text: after removing stopwords (ready for ML models)
    """
    
    print(f"\nProcessing: {pdf_path}")
    print("=" * 50)
    
    # Step 1: Extract
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(raw_text)} characters of raw text")
    
    # Step 2: Clean
    cleaned_text = clean_text(raw_text)
    print(f"✓ Cleaned text: {len(cleaned_text)} characters")
    
    # Step 3: Remove stopwords
    final_text = remove_stopwords(cleaned_text)
    print(f"✓ After stopword removal: {len(final_text.split())} words remaining")
    
    return raw_text, cleaned_text, final_text
     # ============================================================
# FUNCTION 5: Save the cleaned text to a file
# ============================================================
def save_cleaned_text(final_text, output_path):
    """
    Saves the final processed text into a .txt file.
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✓ Saved cleaned text to {output_path}")


# ============================================================
# RUN IT: Test on all your sample resumes
# ============================================================
if __name__ == "__main__":

    import os                      # ← ADD THIS

    os.makedirs("cleaned_resumes", exist_ok=True)   # ← ADD THIS

    resume_files = [
        "sample_resumes/resume1.pdf",
        "sample_resumes/resume2.pdf",
        "sample_resumes/resume3.pdf",
        "sample_resumes/resume4.pdf",
    ]

    for i, resume_path in enumerate(resume_files, 1):   # ← CHANGE THIS LINE
        raw, cleaned, final = process_resume(resume_path)

        output_file = f"cleaned_resumes/resume{i}_cleaned.txt"   # ← ADD THIS
        save_cleaned_text(final, output_file)                    # ← ADD THIS

        print("\n--- RAW TEXT (first 300 characters) ---")
        print(raw[:300])

        print("\n--- CLEANED TEXT (first 300 characters) ---")
        print(cleaned[:300])

        print("\n--- FINAL TEXT after stopword removal (first 300 characters) ---")
        print(final[:300])

        print("\n" + "="*60 + "\n")