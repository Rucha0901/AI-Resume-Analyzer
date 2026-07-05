# similarity_matcher.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import your cleaning functions from Day 3-4
from pdf_extractor import clean_text, remove_stopwords

# Import your skills extractor from Day 5-6
from entity_extractor import extract_skills


# ================================================================
# STEP 1: TEXT PREPARATION
# Prepare resume and JD text before vectorizing
# ================================================================

def prepare_text(raw_text):
    """
    Takes raw text and runs the full cleaning pipeline.
    This is the same cleaning you did on Days 3-4,
    now used here to prepare text for TF-IDF.
    
    Why clean again? Because TF-IDF works best on clean,
    normalized text. Punctuation and stopwords add noise
    that reduces similarity accuracy.
    """
    cleaned = clean_text(raw_text)
    final = remove_stopwords(cleaned)
    return final


# ================================================================
# STEP 2: TF-IDF VECTORIZATION
# Convert text into numerical vectors
# ================================================================

def vectorize_texts(text1, text2):
    """
    Takes two text strings and converts both into TF-IDF vectors.
    
    WHY pass both texts together?
    The TfidfVectorizer needs to see BOTH documents at once
    to build a shared vocabulary and calculate IDF scores.
    IDF is calculated ACROSS documents — so it needs to know
    what words appear in one vs. both vs. neither document.
    
    If you vectorized them separately, they'd have different
    vocabularies and you couldn't compare them.
    
    Returns two vectors (as numpy arrays).
    """
    
    # Create the TF-IDF vectorizer
    # Parameters explained:
    # max_features=5000 → only keep the 5000 most important words
    #                      (prevents memory issues with huge vocabularies)
    # ngram_range=(1,2) → capture both single words AND pairs of words
    #                      e.g., captures "machine" AND "machine learning"
    #                      as separate features. This is crucial for
    #                      multi-word skills like "machine learning"!
    # min_df=1          → include a word even if it appears in only 1 document
    #                      (important since we only have 2 documents)
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True  # applies log to term frequency
                           # prevents very common words from dominating
                           # even after stopword removal
    )
    
    # fit_transform does TWO things:
    # 1. "fit"      → learns the vocabulary and IDF scores from these texts
    # 2. "transform" → converts the texts into TF-IDF vectors
    # 
    # We pass BOTH texts as a list so it builds ONE shared vocabulary
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # tfidf_matrix has shape (2, vocabulary_size)
    # Row 0 = vector for text1 (resume)
    # Row 1 = vector for text2 (job description)
    
    # Let's see what the vocabulary looks like
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n   Vocabulary size: {len(feature_names)} unique terms")
    print(f"   Sample terms: {list(feature_names[:10])}")
    
    return tfidf_matrix, vectorizer


# ================================================================
# STEP 3: COMPUTE COSINE SIMILARITY
# Calculate how similar the two vectors are
# ================================================================

def compute_similarity(tfidf_matrix):
    """
    Takes the TF-IDF matrix (containing both vectors)
    and calculates cosine similarity between them.
    
    Returns a score between 0.0 and 1.0:
    0.0 = completely different
    1.0 = identical
    """
    
    # Extract the two vectors
    resume_vector = tfidf_matrix[0]   # first row = resume
    jd_vector     = tfidf_matrix[1]   # second row = job description
    
    # cosine_similarity expects 2D arrays, so we compare
    # resume_vector against jd_vector
    # The result is a 1x1 matrix, so we extract [0][0] to get the number
    similarity_score = cosine_similarity(resume_vector, jd_vector)[0][0]
    
    return similarity_score


# ================================================================
# STEP 4: FIND MATCHING AND MISSING SKILLS
# This is your custom logic that makes the analyzer actually useful
# ================================================================

def find_skill_gaps(resume_text, jd_text):
    """
    Compares skills found in resume vs skills mentioned in JD.
    
    Returns:
    - matched_skills: skills in BOTH resume and JD (your strengths)
    - missing_skills: skills in JD but NOT in resume (gaps to fill)
    - extra_skills:   skills in resume but NOT in JD (bonus skills)
    
    This is your "substantial custom logic" that goes beyond
    just calling TF-IDF. You're doing semantic skill gap analysis.
    """
    
    # Extract skills from resume
    resume_skills, resume_by_cat = extract_skills(resume_text)
    resume_skills_set = set([s.lower() for s in resume_skills])
    
    # Extract skills from job description
    jd_skills, jd_by_cat = extract_skills(jd_text)
    jd_skills_set = set([s.lower() for s in jd_skills])
    
    # Compare the two sets
    matched_skills = resume_skills_set.intersection(jd_skills_set)
    missing_skills = jd_skills_set - resume_skills_set      # in JD but not resume
    extra_skills   = resume_skills_set - jd_skills_set      # in resume but not JD
    
    return {
        "matched" : sorted(list(matched_skills)),
        "missing" : sorted(list(missing_skills)),
        "extra"   : sorted(list(extra_skills)),
        "jd_total": len(jd_skills_set),
        "resume_total": len(resume_skills_set)
    }


# ================================================================
# STEP 5: CALCULATE SKILL-BASED SCORE
# A second scoring method based purely on skill overlap
# ================================================================

def calculate_skill_score(skill_gaps):
    """
    Calculates a separate score based ONLY on skill matching.
    
    Formula:
    skill_score = (matched skills / total skills in JD) × 100
    
    Example:
    JD requires 10 skills. Resume has 7 of them.
    Skill score = 7/10 × 100 = 70%
    
    This is different from TF-IDF score because:
    - TF-IDF looks at ALL words and their frequencies
    - Skill score looks ONLY at specific skills from your database
    Both together give a more complete picture.
    """
    
    if skill_gaps["jd_total"] == 0:
        return 0.0
    
    matched_count = len(skill_gaps["matched"])
    jd_total      = skill_gaps["jd_total"]
    
    score = (matched_count / jd_total) * 100
    return round(score, 2)


# ================================================================
# STEP 6: FINAL COMBINED SCORE
# Weighted combination of TF-IDF score and skill score
# ================================================================

def calculate_final_score(tfidf_score, skill_score):
    """
    Combines TF-IDF similarity and skill match score
    into ONE final score.
    
    We give more weight to TF-IDF because it considers
    the full context of the text, not just skill keywords.
    But skills are important too, so we don't ignore them.
    
    Weight breakdown (you can adjust these):
    - TF-IDF score: 60% weight
    - Skill score:  40% weight
    """
    
    tfidf_percentage = tfidf_score * 100  # convert 0.78 → 78
    
    final = (0.60 * tfidf_percentage) + (0.40 * skill_score)
    return round(final, 2)


# ================================================================
# STEP 7: THE MASTER FUNCTION
# Puts every step together into one clean function call
# ================================================================

def analyze_match(resume_text, jd_text):
    """
    MAIN FUNCTION: Takes resume text + job description text,
    runs all analysis, returns complete results.
    
    This is what your Streamlit UI will call in the end.
    """
    
    print("\n" + "="*55)
    print("       RESUME vs JOB DESCRIPTION ANALYSIS")
    print("="*55)
    
    # --- STEP A: Prepare texts ---
    print("\n[1/4] Preparing texts...")
    resume_prepared = prepare_text(resume_text)
    jd_prepared     = prepare_text(jd_text)
    
    print(f"   Resume: {len(resume_prepared.split())} words after cleaning")
    print(f"   JD:     {len(jd_prepared.split())} words after cleaning")
    
    # --- STEP B: TF-IDF Vectorization ---
    print("\n[2/4] Building TF-IDF vectors...")
    tfidf_matrix, vectorizer = vectorize_texts(resume_prepared, jd_prepared)
    
    # --- STEP C: Cosine Similarity ---
    print("\n[3/4] Computing similarity...")
    tfidf_score = compute_similarity(tfidf_matrix)
    print(f"   Raw TF-IDF Cosine Score: {tfidf_score:.4f}")
    
    # --- STEP D: Skill Gap Analysis ---
    print("\n[4/4] Analyzing skill gaps...")
    skill_gaps   = find_skill_gaps(resume_text, jd_text)
    skill_score  = calculate_skill_score(skill_gaps)
    final_score  = calculate_final_score(tfidf_score, skill_score)
    
    print(f"   Skill Match Score: {skill_score}%")
    print(f"   Final Combined Score: {final_score}%")
    
    # --- Package all results ---
    results = {
        "tfidf_score"   : round(tfidf_score * 100, 2),
        "skill_score"   : skill_score,
        "final_score"   : final_score,
        "skill_gaps"    : skill_gaps,
        "resume_words"  : len(resume_prepared.split()),
        "jd_words"      : len(jd_prepared.split())
    }
    
    return results


# ================================================================
# STEP 8: PRINT RESULTS IN A READABLE FORMAT
# ================================================================

def print_match_results(results):
    """Displays the analysis results clearly"""
    
    gaps = results["skill_gaps"]
    
    print("\n" + "="*55)
    print("              MATCH RESULTS")
    print("="*55)
    
    # Score display with visual bar
    score = results["final_score"]
    bar_length = int(score / 5)  # max 20 chars for 100%
    bar = "█" * bar_length + "░" * (20 - bar_length)
    
    print(f"\n  FINAL MATCH SCORE:  {score}%")
    print(f"  [{bar}]")
    
    # Score interpretation
    if score >= 80:
        verdict = "EXCELLENT MATCH — Strong candidate!"
    elif score >= 60:
        verdict = "GOOD MATCH — Worth applying!"
    elif score >= 40:
        verdict = "MODERATE MATCH — Consider upskilling"
    else:
        verdict = "WEAK MATCH — Significant gaps exist"
    
    print(f"\n  VERDICT: {verdict}")
    
    print(f"\n  TF-IDF Score  : {results['tfidf_score']}%")
    print(f"  Skill Score   : {results['skill_score']}%")
    
    print(f"\n  SKILLS MATCHED ({len(gaps['matched'])}/{gaps['jd_total']}):")
    if gaps["matched"]:
        for skill in gaps["matched"]:
            print(f"    ✅ {skill}")
    else:
        print("    None found")
    
    print(f"\n  SKILLS MISSING FROM YOUR RESUME:")
    if gaps["missing"]:
        for skill in gaps["missing"]:
            print(f"    ❌ {skill}")
    else:
        print("    None — you have all required skills!")
    
    print(f"\n  BONUS SKILLS YOU HAVE (not in JD):")
    if gaps["extra"]:
        for skill in list(gaps["extra"])[:5]:  # show max 5
            print(f"    ➕ {skill}")
    
    print("\n" + "="*55)