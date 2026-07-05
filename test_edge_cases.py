# test_edge_cases.py
# Test every edge case before your evaluators find them
# Run with: python test_edge_cases.py

import sys
import os

# ================================================================
# TEST FRAMEWORK
# A simple class to track pass/fail results
# ================================================================

class TestRunner:
    """
    Simple test runner to track which tests pass and fail.
    Professional developers use pytest or unittest for this,
    but this simple version is easier to understand.
    """
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, test_name, condition, failure_message=""):
        """
        Checks if condition is True.
        Prints PASS or FAIL with the test name.
        """
        if condition:
            print(f"  ✅ PASS: {test_name}")
            self.passed += 1
        else:
            print(f"  ❌ FAIL: {test_name}")
            if failure_message:
                print(f"         └─ {failure_message}")
            self.failed += 1
            self.errors.append(test_name)
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'='*50}")
        return self.failed == 0  # True if all passed


tr = TestRunner()

print("\n" + "="*50)
print("RUNNING EDGE CASE TESTS")
print("="*50)


# ================================================================
# TEST GROUP 1: TEXT CLEANING
# ================================================================

print("\n📋 GROUP 1: Text Cleaning Tests")
print("-"*40)

from pdf_extractor import clean_text, remove_stopwords

# Test 1: Empty string
result = clean_text("")
tr.test(
    "Clean empty string returns empty string",
    result == "",
    f"Got: '{result}'"
)

# Test 2: Only special characters
result = clean_text("!@#$%^&*()")
tr.test(
    "Clean special-only string returns empty/whitespace",
    result.strip() == "",
    f"Got: '{result}'"
)

# Test 3: Normal text is cleaned properly
result = clean_text("John Doe — Software Engineer at Google, Inc. • Python/Django")
tr.test(
    "Normal text cleaned correctly",
    "john" in result.lower() and "python" in result.lower(),
    f"Got: '{result}'"
)

# Test 4: URL removed from text
result = clean_text("Visit https://linkedin.com/in/johndoe for profile")
tr.test(
    "URLs removed from text",
    "https" not in result and "linkedin.com" not in result,
    f"Got: '{result}'"
)

# Test 5: Email removed from text
result = clean_text("Contact me at john.doe@gmail.com for details")
tr.test(
    "Emails removed from cleaned text",
    "@gmail.com" not in result,
    f"Got: '{result}'"
)

# Test 6: Stopwords removed
result = remove_stopwords("the quick brown fox jumped over the lazy dog")
tr.test(
    "Stopwords removed correctly",
    "the" not in result.split() and "over" not in result.split(),
    f"Got: '{result}'"
)

# Test 7: Meaningful words survive stopword removal
result = remove_stopwords("experienced python developer machine learning")
tr.test(
    "Meaningful words survive stopword removal",
    "python" in result and "machine" in result,
    f"Got: '{result}'"
)


# ================================================================
# TEST GROUP 2: ENTITY EXTRACTION
# ================================================================

print("\n📋 GROUP 2: Entity Extraction Tests")
print("-"*40)

from entity_extractor import extract_email, extract_phone, extract_skills

# Test 8: Valid email detection
email = extract_email("Contact: john.doe@gmail.com")
tr.test(
    "Valid email detected",
    email == "john.doe@gmail.com",
    f"Got: '{email}'"
)

# Test 9: Email with subdomain detected
email = extract_email("Email: user@company.co.in")
tr.test(
    "Email with subdomain detected",
    email is not None and "@" in email,
    f"Got: '{email}'"
)

# Test 10: No email returns None
email = extract_email("No email in this text at all")
tr.test(
    "No email returns None",
    email is None,
    f"Got: '{email}'"
)

# Test 11: Valid phone detected
phone = extract_phone("Call me at +91 9876543210")
tr.test(
    "Valid phone number detected",
    phone is not None,
    f"Got: '{phone}'"
)

# Test 12: Short number not mistaken as phone
phone = extract_phone("I have 123 items")
tr.test(
    "Short number not mistaken as phone",
    phone is None,
    f"Got: '{phone}'"
)

# Test 13: Python skill detected
skills, _ = extract_skills("experienced python developer with django and sql")
skills_lower = [s.lower() for s in skills]
tr.test(
    "Python skill detected from text",
    "python" in skills_lower,
    f"Got skills: {skills_lower}"
)

# Test 14: Multi-word skill detected
skills, _ = extract_skills("proficient in machine learning and deep learning")
skills_lower = [s.lower() for s in skills]
tr.test(
    "Multi-word skill 'machine learning' detected",
    "machine learning" in skills_lower,
    f"Got skills: {skills_lower}"
)

# Test 15: 'java' doesn't match 'javascript'
skills, _ = extract_skills("expert in javascript and react")
skills_lower = [s.lower() for s in skills]
tr.test(
    "Word boundary prevents 'java' matching in 'javascript'",
    "java" not in skills_lower,
    f"Got skills: {skills_lower}"
)

# Test 16: Empty text returns empty skills
skills, _ = extract_skills("")
tr.test(
    "Empty text returns empty skills list",
    skills == [],
    f"Got: {skills}"
)


# ================================================================
# TEST GROUP 3: SIMILARITY SCORING
# ================================================================

print("\n📋 GROUP 3: Similarity Scoring Tests")
print("-"*40)

from similarity_matcher import analyze_match

# Test 17: Identical texts score very high
results = analyze_match(
    "Python developer machine learning data science tensorflow",
    "Python developer machine learning data science tensorflow"
)
tr.test(
    "Identical texts score above 90%",
    results["tfidf_score"] > 90,
    f"Got: {results['tfidf_score']}%"
)

# Test 18: Completely different texts score low
results = analyze_match(
    "Python developer machine learning tensorflow data science",
    "Chef cooking recipe ingredients restaurant kitchen culinary"
)
tr.test(
    "Completely different texts score below 30%",
    results["final_score"] < 30,
    f"Got: {results['final_score']}%"
)

# Test 19: Score is between 0 and 100
results = analyze_match("python java sql", "python machine learning")
tr.test(
    "Score is always between 0 and 100",
    0 <= results["final_score"] <= 100,
    f"Got: {results['final_score']}%"
)

# Test 20: Matched skills are subset of both
results = analyze_match(
    "python sql machine learning",
    "python sql aws kubernetes"
)
matched = results["skill_gaps"]["matched"]
tr.test(
    "Matched skills appear in both resume and JD",
    all(skill in ["python", "sql"] for skill in matched),
    f"Got matched: {matched}"
)

# Test 21: Missing skills are in JD but not resume
missing = results["skill_gaps"]["missing"]
tr.test(
    "Missing skills are from JD only",
    len(missing) > 0,
    f"Got missing: {missing}"
)


# ================================================================
# TEST GROUP 4: VALIDATOR FUNCTIONS
# ================================================================

print("\n📋 GROUP 4: Input Validation Tests")
print("-"*40)

from utils import validate_job_description, validate_extracted_text

# Test 22: Empty JD fails validation
is_valid, error, _ = validate_job_description("")
tr.test(
    "Empty JD fails validation",
    not is_valid and error is not None,
    f"is_valid={is_valid}, error={error}"
)

# Test 23: Very short JD fails validation
is_valid, error, _ = validate_job_description("Python developer")
tr.test(
    "Too-short JD fails validation",
    not is_valid,
    f"is_valid={is_valid}"
)

# Test 24: Good JD passes validation
good_jd = "We need a Python developer with 3+ years experience in Django, REST APIs, PostgreSQL, AWS, and Docker."
is_valid, error, _ = validate_job_description(good_jd)
tr.test(
    "Good JD passes validation",
    is_valid and error is None,
    f"is_valid={is_valid}, error={error}"
)

# Test 25: Garbage text fails extracted text validation
is_valid, _ = validate_extracted_text("abc def")
tr.test(
    "Too-short extracted text fails validation",
    not is_valid,
    f"is_valid={is_valid}"
)

# Test 26: Good resume text passes validation
good_text = """
John Doe is an experienced software engineer with 5 years of experience
in Python development, machine learning, and data science.
He has worked at Google and Amazon on large-scale distributed systems.
His skills include Python, TensorFlow, SQL, Docker, and Kubernetes.
Education: B.Tech Computer Science from IIT Bombay.
Email: john@gmail.com, Phone: 9876543210
""" * 3  # repeat to make it long enough
is_valid, _ = validate_extracted_text(good_text)
tr.test(
    "Good resume text passes validation",
    is_valid,
    f"is_valid={is_valid}"
)


# ================================================================
# TEST GROUP 5: MODEL LOADING
# ================================================================

print("\n📋 GROUP 5: Model Loading Tests")
print("-"*40)

from utils import load_all_models

models = load_all_models()

# Test 27: Models load without crashing
tr.test(
    "load_all_models() runs without exception",
    True  # if we got here without exception, it passed
)

# Test 28: If models exist, they loaded successfully
if os.path.exists("models/logistic_regression_model.pkl"):
    tr.test(
        "Classifier model loaded successfully",
        models["classifier"] is not None,
        f"Error: {models.get('error')}"
    )
    
    tr.test(
        "Vectorizer loaded successfully",
        models["vectorizer"] is not None,
        f"Error: {models.get('error')}"
    )
    
    tr.test(
        "Label encoder loaded successfully",
        models["label_encoder"] is not None,
        f"Error: {models.get('error')}"
    )
else:
    print("  ⚠️  SKIP: Model files not found (run train_classifier.py first)")


# ================================================================
# PRINT FINAL SUMMARY
# ================================================================

all_passed = tr.summary()

if all_passed:
    print("\n🎉 All tests passed! Your code is robust.")
else:
    print("\n⚠️  Some tests failed. Fix those issues before submission.")
    sys.exit(1)  # exit with error code so you know tests failed