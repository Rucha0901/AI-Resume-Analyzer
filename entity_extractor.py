# entity_extractor.py

import re           # for regex patterns
import spacy        # for NER (Names and Organizations)
from skills_database import get_all_skills, get_skills_by_category

# Load spaCy's pretrained English model
# This line loads the "brain" that spaCy will use to recognize
# names, places, organizations, dates in text
# (You downloaded this on Day 1 with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


# ================================================================
# PART 1: EXTRACT CONTACT INFORMATION USING REGEX
# ================================================================

def extract_email(text):
    """
    Uses regex to find email addresses in text.
    
    The pattern means:
    - [a-zA-Z0-9._%+-]+  : username (letters, digits, dots, underscores etc.)
    - @                   : literal @ symbol
    - [a-zA-Z0-9.-]+      : domain name (gmail, yahoo, company etc.)
    - \.                  : literal dot
    - [a-zA-Z]{2,}        : extension (com, in, org etc.) - at least 2 letters
    """
    
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # re.findall() searches the ENTIRE text and returns ALL matches as a list
    emails = re.findall(email_pattern, text)
    
    # Return first email found (resumes usually have just one)
    # If no email found, return None
    return emails[0] if emails else None


def extract_phone(text):
    """
    Uses regex to find phone numbers in text.
    
    Phone numbers come in MANY formats:
    - +91 98765 43210   (Indian with country code)
    - 9876543210        (plain 10 digits)
    - 098-765-4321      (with hyphens)
    - (098) 765-4321    (with brackets)
    - +1-555-123-4567   (US format)
    
    We need a pattern flexible enough to catch all of these.
    """
    
    # This pattern handles all the common formats:
    # [\+]?        : optional + sign at start
    # [(]?         : optional opening bracket
    # [0-9]{1,4}   : 1 to 4 digits (country code like 91, 1, 44)
    # [)]?         : optional closing bracket  
    # [-\s\.]?     : optional separator (hyphen, space, or dot)
    # [0-9]{3,5}   : 3 to 5 digits
    # [-\s\.]?     : optional separator again
    # [0-9]{3,4}   : 3 to 4 digits
    # [-\s\.]?     : optional separator
    # [0-9]{3,4}   : final group of digits
    
    phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{3,4}[-\s\.]?[0-9]{3,4}'
    
    phones = re.findall(phone_pattern, text)
    
    # Filter results: must be at least 10 digits total
    # (removes short false matches like years "2020" which also match digit patterns)
    valid_phones = []
    for phone in phones:
        # Count only the digits in what we found
        digits_only = re.sub(r'\D', '', phone)  # remove non-digits
        if len(digits_only) >= 10:              # valid phone has at least 10 digits
            valid_phones.append(phone.strip())
    
    return valid_phones[0] if valid_phones else None


def extract_linkedin(text):
    """
    Looks for LinkedIn profile URLs.
    Pattern: linkedin.com/in/ followed by the username
    """
    
    linkedin_pattern = r'linkedin\.com/in/[a-zA-Z0-9\-_]+'
    
    matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
    
    return matches[0] if matches else None


# ================================================================
# PART 2: EXTRACT NAME AND ORGANIZATIONS USING spaCy NER
# ================================================================

def extract_name_with_spacy(text):
    """
    Uses spaCy's pretrained NER model to find person names.
    
    How spaCy NER works (simple explanation):
    spaCy was trained on millions of documents where humans labeled
    every word: "this is a PERSON", "this is an ORG", "this is a DATE".
    spaCy learned statistical patterns from all those labels.
    Now when it sees new text, it predicts labels for each word/phrase.
    
    Entity types we care about:
    - PERSON  : person names ("Amit Kumar", "Jane Doe")
    - ORG     : organizations ("Google", "Infosys", "IIT Mumbai")
    - GPE     : countries, cities ("Mumbai", "India", "USA")
    - DATE    : dates ("2020-2022", "January 2019")
    """
    
    # Run spaCy on the text - this does ALL the NLP processing
    # (tokenizing, POS tagging, NER, dependency parsing) in one line
    doc = nlp(text[:1000])  # Only look at first 1000 chars for names
                             # Names are almost always at the top of a resume
    
    persons = []
    
    # doc.ents gives us all the "entities" spaCy found
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            # Clean the name: remove extra spaces
            name = entity.text.strip()
            
            # Basic validation: names usually have 2-4 words
            # and don't contain digits
            words = name.split()
            if 1 < len(words) <= 4 and not any(char.isdigit() for char in name):
                persons.append(name)
    
    # Return the first person found (most likely the resume owner's name)
    return persons[0] if persons else None


def extract_organizations_with_spacy(text):
    """
    Uses spaCy NER to find company/organization names.
    These are the companies the person has worked at.
    """
    
    doc = nlp(text)
    
    organizations = []
    seen = set()  # track what we've already added (avoid duplicates)
    
    for entity in doc.ents:
        if entity.label_ == "ORG":
            org_name = entity.text.strip()
            
            # Avoid duplicates (case-insensitive check)
            if org_name.lower() not in seen and len(org_name) > 2:
                organizations.append(org_name)
                seen.add(org_name.lower())
    
    return organizations


# ================================================================
# PART 3: EXTRACT EDUCATION USING REGEX
# (spaCy misses education details, regex is better here)
# ================================================================

def extract_education(text):
    """
    Finds education qualifications using regex pattern matching.
    
    We look for common degree abbreviations and keywords.
    """
    
    education_found = []
    
    # Common degree patterns - | means OR between options
    degree_patterns = [
        r'\b(B\.?Tech|BE|B\.?E|Bachelor of Technology|Bachelor of Engineering)\b',
        r'\b(M\.?Tech|ME|M\.?E|Master of Technology|Master of Engineering)\b',
        r'\b(BCA|MCA|B\.?C\.?A|M\.?C\.?A)\b',
        r'\b(BSc|B\.?Sc|Bachelor of Science)\b',
        r'\b(MSc|M\.?Sc|Master of Science)\b',
        r'\b(MBA|Master of Business Administration)\b',
        r'\b(BBA|Bachelor of Business Administration)\b',
        r'\b(PhD|Ph\.?D|Doctorate|Doctor of Philosophy)\b',
        r'\b(B\.?Com|Bachelor of Commerce)\b',
        r'\b(Diploma|Certificate|Certification)\b',
        r'\b(10th|12th|SSC|HSC|Matric|Intermediate)\b',
    ]
    
    for pattern in degree_patterns:
        # re.IGNORECASE makes matching case-insensitive
        # so "btech" matches "BTech" matches "B.Tech"
        matches = re.findall(pattern, text, re.IGNORECASE)
        education_found.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_education = []
    for item in education_found:
        if item.lower() not in seen:
            unique_education.append(item)
            seen.add(item.lower())
    
    return unique_education


# ================================================================
# PART 4: EXTRACT SKILLS — THE MOST IMPORTANT PART
# ================================================================

def extract_skills(text):
    """
    Compares the resume text against your master skills database.
    
    METHOD: For each skill in your database, check if it appears
    in the resume text. If yes, it's a found skill.
    
    Two types of skills require different matching:
    - Single word skills: "python", "sql", "java" 
      → use word boundary matching (\b) so "java" doesn't match "javascript"
    - Multi-word skills: "machine learning", "deep learning", "react native"
      → search for the whole phrase
    """
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    all_skills = get_all_skills()
    skills_by_category = get_skills_by_category()
    
    found_skills = []           # flat list of all found skills
    found_by_category = {}      # organized by category
    
    # Check EACH skill in your database
    for skill in all_skills:
        skill_lower = skill.lower()
        
        if " " in skill_lower:
            # MULTI-WORD SKILL: search for the exact phrase
            # e.g., "machine learning" should appear as-is in the text
            if skill_lower in text_lower:
                found_skills.append(skill)
        else:
            # SINGLE WORD SKILL: use word boundaries (\b)
            # \b means "word boundary" - the edge between a word and non-word character
            # Pattern \bpython\b matches "python" but NOT "pythonista" or "xpython"
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            
            # re.escape() is important: it makes special chars in skill names
            # treated as literal characters (e.g., "c++" becomes "c\+\+" in regex)
            
            if re.search(pattern, text_lower):
                found_skills.append(skill)
    
    # Also organize found skills by category
    for category, skills_list in skills_by_category.items():
        category_found = []
        for skill in skills_list:
            if skill in [s.lower() for s in found_skills]:
                category_found.append(skill)
        if category_found:
            found_by_category[category] = category_found
    
    return found_skills, found_by_category


# ================================================================
# PART 5: EXTRACT YEARS OF EXPERIENCE
# ================================================================

def extract_experience_years(text):
    """
    Tries to estimate total years of experience from the resume.
    
    Looks for patterns like:
    - "5 years of experience"
    - "3+ years"
    - "2-4 years"
    """
    
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)-(\d+)\s*years?',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Return the first number found
            first_match = matches[0]
            if isinstance(first_match, tuple):
                return int(first_match[0])  # take lower bound of range
            return int(first_match)
    
    return None  # couldn't determine years


# ================================================================
# PART 6: THE MASTER FUNCTION — PUT IT ALL TOGETHER
# ================================================================

def extract_all_entities(raw_text, cleaned_text):
    """
    Runs ALL extraction functions and returns a neat dictionary
    with everything we found.
    
    raw_text   = original extracted PDF text (needed for NER, email, phone)
    cleaned_text = lowercased, cleaned text (needed for skills matching)
    """
    
    print("\nExtracting entities...")
    
    # Run every extractor
    email       = extract_email(raw_text)
    phone       = extract_phone(raw_text)
    linkedin    = extract_linkedin(raw_text)
    name        = extract_name_with_spacy(raw_text)
    orgs        = extract_organizations_with_spacy(raw_text)
    education   = extract_education(raw_text)
    skills, skills_by_cat = extract_skills(cleaned_text)
    experience  = extract_experience_years(raw_text)
    
    # Package everything into a dictionary
    results = {
        "name"              : name,
        "email"             : email,
        "phone"             : phone,
        "linkedin"          : linkedin,
        "organizations"     : orgs,
        "education"         : education,
        "skills"            : skills,
        "skills_by_category": skills_by_cat,
        "experience_years"  : experience,
        "total_skills_found": len(skills)
    }
    
    return results


def print_results(results):
    """Nicely prints the extraction results"""
    
    print("\n" + "="*55)
    print("         ENTITY EXTRACTION RESULTS")
    print("="*55)
    
    print(f"\n👤  NAME          : {results['name'] or 'Not found'}")
    print(f"📧  EMAIL         : {results['email'] or 'Not found'}")
    print(f"📱  PHONE         : {results['phone'] or 'Not found'}")
    print(f"🔗  LINKEDIN      : {results['linkedin'] or 'Not found'}")
    print(f"⏱️   EXPERIENCE    : {str(results['experience_years']) + ' years' if results['experience_years'] else 'Not found'}")
    
    print(f"\n🎓  EDUCATION:")
    if results['education']:
        for edu in results['education']:
            print(f"    - {edu}")
    else:
        print("    Not found")
    
    print(f"\n🏢  ORGANIZATIONS WORKED AT:")
    if results['organizations']:
        for org in results['organizations'][:5]:  # show max 5
            print(f"    - {org}")
    else:
        print("    Not found")
    
    print(f"\n⚡  SKILLS FOUND ({results['total_skills_found']} total):")
    for category, skills in results['skills_by_category'].items():
        print(f"\n    [{category.upper().replace('_', ' ')}]")
        print(f"    {', '.join(skills)}")
    
    print("\n" + "="*55)