# app.py
# Run with: streamlit run app.py

import streamlit as st
import joblib
import tempfile
import os
import re
import time

# Import all your modules from previous days
from pdf_extractor import extract_text_from_pdf, clean_text, remove_stopwords
from entity_extractor import extract_all_entities
from similarity_matcher import analyze_match
from train_classifier import clean_resume_text, predict_resume_category


# ================================================================
# SECTION 1: PAGE CONFIGURATION
# Must be the FIRST streamlit command in your script
# ================================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",          # "wide" uses full browser width
                            # "centered" uses a narrow centered column
    initial_sidebar_state="expanded"
)


# ================================================================
# SECTION 2: CUSTOM STYLING
# Streamlit allows you to inject CSS for visual customization
# ================================================================

st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Score display box */
    .score-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    
    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Missing skill tags */
    .skill-tag-missing {
        display: inline-block;
        background-color: #fce4ec;
        color: #c62828;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Matched skill tags */
    .skill-tag-matched {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Section cards */
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Bigger button */
    .stButton button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)
# unsafe_allow_html=True means "yes, I know this is HTML, allow it"


# ================================================================
# SECTION 3: LOAD TRAINED MODELS
# Use st.cache_resource so models load ONCE and stay in memory
# Without this, models would reload every time user interacts
# ================================================================

@st.cache_resource
def load_models():
    """
    Loads trained ML models from disk.
    
    @st.cache_resource is a decorator (special instruction above function).
    It tells Streamlit: "run this function only once, then cache the result.
    Every future call to this function returns the cached result immediately."
    
    This is critical for performance — loading models takes 2-3 seconds.
    Without caching, every button click would reload models = very slow app.
    """
    
    try:
        model     = joblib.load("models/logistic_regression_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        le        = joblib.load("models/label_encoder.pkl")
        return model, vectorizer, le
    
    except FileNotFoundError:
        # Models don't exist yet — user hasn't trained them
        return None, None, None


# Load models when app starts
classifier_model, tfidf_vectorizer, label_encoder = load_models()


# ================================================================
# SECTION 4: HELPER FUNCTIONS
# ================================================================

def display_skill_tags(skills, tag_type="normal"):
    """
    Displays skills as colored pill/badge tags.
    
    tag_type options:
    "normal"  → blue tags (resume skills)
    "matched" → green tags (skills that match JD)
    "missing" → red tags (skills missing from resume)
    """
    
    if not skills:
        st.write("None found")
        return
    
    css_class = {
        "normal" : "skill-tag",
        "matched": "skill-tag-matched",
        "missing": "skill-tag-missing"
    }.get(tag_type, "skill-tag")
    
    # Build HTML string of all skill tags
    tags_html = ""
    for skill in skills:
        tags_html += f'<span class="{css_class}">{skill}</span> '
    
    st.markdown(tags_html, unsafe_allow_html=True)


def display_score_gauge(score):
    """
    Creates a visual progress bar for the match score.
    Changes color based on score:
    - Red:    0-40%  (poor match)
    - Orange: 40-60% (moderate)
    - Yellow: 60-75% (good)
    - Green:  75%+   (excellent)
    """
    
    if score >= 75:
        color = "#4caf50"    # green
        verdict = "Excellent Match! 🌟"
        advice = "Strong profile for this role. Apply with confidence!"
    elif score >= 60:
        color = "#ff9800"    # orange
        verdict = "Good Match 👍"
        advice = "Solid profile. Address the missing skills to strengthen your application."
    elif score >= 40:
        color = "#ff5722"    # deep orange
        verdict = "Moderate Match ⚠️"
        advice = "Consider upskilling in the missing areas before applying."
    else:
        color = "#f44336"    # red
        verdict = "Weak Match ❌"
        advice = "Significant skill gaps exist. This role may need more preparation."
    
    # HTML for the score display
    # We build this manually so we can control colors dynamically
    st.markdown(f"""
        <div class="score-box">
            <h1 style="font-size: 72px; margin: 0;">{score}%</h1>
            <h3 style="margin: 10px 0;">{verdict}</h3>
            <p style="opacity: 0.9; font-size: 16px;">{advice}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Streamlit's built-in progress bar
    st.progress(score / 100)


def extract_contact_info_display(entities):
    """Displays extracted contact info in a clean layout"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Name**")
        st.write(entities.get('name') or "Not detected")
    
    with col2:
        st.markdown("**📧 Email**")
        st.write(entities.get('email') or "Not detected")
    
    with col3:
        st.markdown("**📱 Phone**")
        st.write(entities.get('phone') or "Not detected")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("**🔗 LinkedIn**")
        st.write(entities.get('linkedin') or "Not detected")
    
    with col5:
        st.markdown("**⏱️ Experience**")
        exp = entities.get('experience_years')
        st.write(f"{exp} years" if exp else "Not detected")


# ================================================================
# SECTION 5: SIDEBAR
# The sidebar shows on the left — good for settings and info
# ================================================================

with st.sidebar:
    
    st.image("https://img.icons8.com/fluency/96/resume.png", width=80)
    st.title("AI Resume Analyzer")
    st.markdown("---")
    
    st.markdown("""
    ### How to use:
    1. 📤 Upload your resume PDF
    2. 📋 Paste the job description
    3. 🔍 Click **Analyze Resume**
    4. 📊 Review your results
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### About this tool:
    This analyzer uses:
    - **TF-IDF** for text vectorization
    - **Cosine Similarity** for matching
    - **spaCy NER** for entity extraction
    - **Logistic Regression** for job category prediction
    """)
    
    st.markdown("---")
    
    # Model status indicator
    st.markdown("### Model Status:")
    if classifier_model is not None:
        st.success("✅ Classifier loaded")
    else:
        st.error("❌ Classifier not found")
        st.caption("Run train_classifier.py first")


# ================================================================
# SECTION 6: MAIN PAGE HEADER
# ================================================================

st.title("📄 AI Resume Analyzer")
st.markdown(
    "Upload your resume and paste a job description to get an instant "
    "**match score**, **skill gap analysis**, and **career suggestions**."
)
st.markdown("---")


# ================================================================
# SECTION 7: INPUT SECTION
# Two columns: left = resume upload, right = job description
# ================================================================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Upload Your Resume")
    
    uploaded_resume = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],                   # only allow PDF files
        help="Upload your resume in PDF format. Max size: 10MB"
    )
    
    if uploaded_resume is not None:
        # Show file details after upload
        file_size_kb = uploaded_resume.size / 1024
        st.success(f"✅ File uploaded: **{uploaded_resume.name}**")
        st.caption(f"Size: {file_size_kb:.1f} KB")
    else:
        # Show a placeholder message when nothing uploaded
        st.info("👆 Please upload your resume PDF")


with col_right:
    st.subheader("📋 Job Description")
    
    job_description = st.text_area(
        label="Paste the job description here:",
        height=250,
        placeholder="""Example:
We are looking for a Data Scientist with:
- Strong Python programming skills
- Experience in Machine Learning and Deep Learning
- Proficiency in TensorFlow or PyTorch
- SQL and data analysis skills
- AWS or GCP cloud experience
- Good communication skills
        """,
        help="Copy and paste the complete job description from any job portal"
    )
    
    # Show word count as user types
    if job_description:
        word_count = len(job_description.split())
        st.caption(f"Word count: {word_count}")
        
        if word_count < 30:
            st.warning("⚠️ Job description seems short. More detail = better analysis.")


# ================================================================
# SECTION 8: ANALYZE BUTTON
# ================================================================

st.markdown("---")

analyze_button = st.button(
    "🔍 Analyze Resume",
    help="Click to start the analysis"
)


# ================================================================
# SECTION 9: VALIDATION CHECKS
# Before running analysis, make sure inputs are valid
# ================================================================

if analyze_button:
    
    # Check: Was a PDF uploaded?
    if uploaded_resume is None:
        st.error("❌ Please upload your resume PDF first!")
        st.stop()   # st.stop() halts execution here — nothing below runs
    
    # Check: Was job description entered?
    if not job_description or len(job_description.strip()) < 20:
        st.error("❌ Please enter a job description (at least 20 characters)!")
        st.stop()
    
    # Check: Are models loaded?
    if classifier_model is None:
        st.error("❌ ML models not found! Please run train_classifier.py first.")
        st.stop()
    
    
    # ================================================================
    # SECTION 10: RUN THE ANALYSIS WITH PROGRESS TRACKING
    # ================================================================
    
    st.markdown("---")
    st.subheader("⚙️ Analysis in Progress...")
    
    # Create a progress bar that we'll update as each step completes
    progress_bar = st.progress(0)
    status_text  = st.empty()    # empty() creates a placeholder we can update
    
    
    # ── STEP 1: Save uploaded PDF to temp file ─────────────────
    # Streamlit gives us a file object, but pdfplumber needs a file PATH
    # We save it temporarily to disk, then read it
    
    status_text.write("📥 Step 1/5: Reading uploaded PDF...")
    progress_bar.progress(10)
    time.sleep(0.3)  # small pause so user can read the status
    
    # tempfile creates a temporary file that gets automatically
    # deleted when we're done with it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_resume.read())
        tmp_path = tmp_file.name
    
    # Extract text from the saved temp file
    raw_text = extract_text_from_pdf(tmp_path)
    
    # Clean up temp file from disk
    os.unlink(tmp_path)
    
    if not raw_text or len(raw_text.strip()) < 50:
        st.error("❌ Could not extract text from this PDF. It might be a scanned/image PDF.")
        st.info("💡 Try a PDF that contains selectable text (not a scanned photo).")
        st.stop()
    
    
    # ── STEP 2: Clean text ─────────────────────────────────────
    status_text.write("🧹 Step 2/5: Cleaning and preprocessing text...")
    progress_bar.progress(25)
    time.sleep(0.3)
    
    cleaned_resume = clean_text(raw_text)
    final_resume   = remove_stopwords(cleaned_resume)
    
    
    # ── STEP 3: Extract entities ───────────────────────────────
    status_text.write("🔍 Step 3/5: Extracting entities (name, skills, education)...")
    progress_bar.progress(45)
    time.sleep(0.3)
    
    entities = extract_all_entities(raw_text, cleaned_resume)
    
    
    # ── STEP 4: Calculate match score ──────────────────────────
    status_text.write("📊 Step 4/5: Calculating match score...")
    progress_bar.progress(70)
    time.sleep(0.3)
    
    match_results = analyze_match(raw_text, job_description)
    
    
    # ── STEP 5: Predict job category ───────────────────────────
    status_text.write("🤖 Step 5/5: Predicting job category with ML model...")
    progress_bar.progress(90)
    time.sleep(0.3)
    
    category_result = predict_resume_category(
        raw_text,
        classifier_model,
        tfidf_vectorizer,
        label_encoder
    )
    
    # All steps done!
    progress_bar.progress(100)
    status_text.write("✅ Analysis complete!")
    time.sleep(0.5)
    
    # Clear the progress indicators — clean them up before showing results
    progress_bar.empty()
    status_text.empty()
    
    
    # ================================================================
    # SECTION 11: DISPLAY RESULTS
    # ================================================================
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    
    # ── RESULT BLOCK 1: MATCH SCORE ────────────────────────────
    
    st.subheader("🎯 Overall Match Score")
    display_score_gauge(match_results["final_score"])
    
    # Three metric cards side by side
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(
            label="🔤 TF-IDF Score",
            value=f"{match_results['tfidf_score']}%",
            help="Similarity based on full text content"
        )
    
    with m2:
        st.metric(
            label="⚡ Skill Match Score",
            value=f"{match_results['skill_score']}%",
            help="Skills from your database that match JD"
        )
    
    with m3:
        skill_gaps = match_results["skill_gaps"]
        st.metric(
            label="📋 Skills Matched",
            value=f"{len(skill_gaps['matched'])}/{skill_gaps['jd_total']}",
            help="Number of JD skills found in your resume"
        )
    
    
    # ── RESULT BLOCK 2: CONTACT INFO ───────────────────────────
    
    st.markdown("---")
    st.subheader("👤 Extracted Contact Information")
    
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        extract_contact_info_display(entities)
        st.markdown('</div>', unsafe_allow_html=True)
    
    
    # ── RESULT BLOCK 3: SKILLS ANALYSIS ────────────────────────
    
    st.markdown("---")
    st.subheader("⚡ Skills Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "✅ Matched Skills",
        "❌ Missing Skills",
        "➕ All Resume Skills",
        "📂 Skills by Category"
    ])
    
    # TAB 1: Skills that appear in BOTH resume and JD
    with tab1:
        st.markdown(
            "These skills from the **job description** were found in your resume:"
        )
        display_skill_tags(skill_gaps["matched"], tag_type="matched")
        
        if skill_gaps["matched"]:
            st.success(
                f"Great job! You have **{len(skill_gaps['matched'])}** "
                f"out of **{skill_gaps['jd_total']}** required skills."
            )
    
    # TAB 2: Skills in JD but NOT in resume
    with tab2:
        st.markdown(
            "These skills are mentioned in the **job description** "
            "but were **not found** in your resume:"
        )
        display_skill_tags(skill_gaps["missing"], tag_type="missing")
        
        if skill_gaps["missing"]:
            st.warning(
                f"⚠️ You're missing **{len(skill_gaps['missing'])}** skills. "
                f"Adding these to your resume or learning them would boost your score."
            )
        else:
            st.success("🌟 Amazing! You have all the required skills!")
    
    # TAB 3: All skills found in the resume
    with tab3:
        st.markdown("All skills detected in your resume:")
        display_skill_tags(entities["skills"], tag_type="normal")
        st.caption(f"Total: {entities['total_skills_found']} skills detected")
    
    # TAB 4: Resume skills organized by category
    with tab4:
        if entities["skills_by_category"]:
            for category, cat_skills in entities["skills_by_category"].items():
                with st.expander(
                    f"📁 {category.replace('_', ' ').title()} ({len(cat_skills)} skills)"
                ):
                    display_skill_tags(cat_skills, tag_type="normal")
        else:
            st.info("No categorized skills found.")
    
    
    # ── RESULT BLOCK 4: EDUCATION & ORGANIZATIONS ──────────────
    
    st.markdown("---")
    
    edu_col, org_col = st.columns(2)
    
    with edu_col:
        st.subheader("🎓 Education Detected")
        
        if entities["education"]:
            for edu in entities["education"]:
                st.markdown(f"• {edu}")
        else:
            st.info("No education details detected")
    
    with org_col:
        st.subheader("🏢 Organizations Detected")
        
        if entities["organizations"]:
            for org in entities["organizations"][:6]:
                st.markdown(f"• {org}")
        else:
            st.info("No organizations detected")
    
    
    # ── RESULT BLOCK 5: ML PREDICTION ─────────────────────────
    
    st.markdown("---")
    st.subheader("🤖 Job Category Prediction (ML Model)")
    st.caption(
        "Our Logistic Regression model, trained on 2484 labeled resumes, "
        "predicts which job category your resume belongs to."
    )
    
    pred_col1, pred_col2 = st.columns([1, 2])
    
    with pred_col1:
        st.metric(
            label="Predicted Category",
            value=category_result["predicted_category"],
        )
        st.metric(
            label="Model Confidence",
            value=f"{category_result['confidence']}%"
        )
    
    with pred_col2:
        st.markdown("**Top 3 Predictions:**")
        
        for i, pred in enumerate(category_result["top3_predictions"]):
            # Display each prediction as a labeled progress bar
            prob = pred["probability"]
            cat  = pred["category"]
            
            bar_label = f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} {cat}"
            st.write(bar_label)
            st.progress(prob / 100)
            st.caption(f"{prob}% confidence")
    
    
    # ── RESULT BLOCK 6: SUGGESTIONS ────────────────────────────
    
    st.markdown("---")
    st.subheader("💡 Personalized Suggestions")
    
    # Generate suggestions based on the actual results
    # This is YOUR custom logic — not copied from an API
    
    suggestions = []
    score = match_results["final_score"]
    missing = skill_gaps["missing"]
    
    # Suggestion logic based on score
    if score < 40:
        suggestions.append({
            "icon": "🎯",
            "title": "Major Skill Gap",
            "text": (
                f"Your resume matches only {score}% of this job description. "
                f"Consider whether this role aligns with your current experience level."
            )
        })
    
    # Suggestion based on missing skills
    if len(missing) > 0:
        top_missing = ", ".join(list(missing)[:4])
        suggestions.append({
            "icon": "📚",
            "title": "Skills to Add or Learn",
            "text": (
                f"The JD requires these skills you're missing: **{top_missing}**. "
                f"Add them to your resume if you have them, or consider learning them."
            )
        })
    
    # Suggestion based on contact info
    if not entities["email"]:
        suggestions.append({
            "icon": "📧",
            "title": "Add Email Address",
            "text": (
                "No email address was detected in your resume. "
                "Ensure your contact details are clearly visible at the top."
            )
        })
    
    if not entities["linkedin"]:
        suggestions.append({
            "icon": "🔗",
            "title": "Add LinkedIn Profile",
            "text": (
                "No LinkedIn URL detected. Adding your LinkedIn profile URL "
                "significantly improves your resume's credibility."
            )
        })
    
    # Suggestion based on experience
    if not entities["experience_years"]:
        suggestions.append({
            "icon": "⏱️",
            "title": "Mention Years of Experience",
            "text": (
                "Consider adding a clear statement of your total years of experience "
                "(e.g., '3+ years of experience in software development')."
            )
        })
    
    # Suggestion based on skills count
    if entities["total_skills_found"] < 8:
        suggestions.append({
            "icon": "⚡",
            "title": "Add More Skills",
            "text": (
                f"Only {entities['total_skills_found']} skills were detected. "
                "A dedicated 'Skills' section with 10-15 relevant skills "
                "significantly improves both ATS scores and human readability."
            )
        })
    
    # Category mismatch suggestion
    predicted = category_result["predicted_category"]
    confidence = category_result["confidence"]
    
    if confidence < 60:
        suggestions.append({
            "icon": "🗂️",
            "title": "Clarify Your Profile",
            "text": (
                f"Your resume was classified as **{predicted}** with only "
                f"{confidence}% confidence, suggesting your profile is mixed. "
                "Focus your resume on one clear job role for better targeting."
            )
        })
    
    # If everything is great
    if score >= 75 and len(missing) <= 2:
        suggestions.append({
            "icon": "🌟",
            "title": "Strong Application — Final Polish",
            "text": (
                "Excellent match! Make sure your bullet points contain "
                "measurable achievements (e.g., 'Improved API response time by 40%' "
                "instead of just 'Worked on API optimization')."
            )
        })
    
    # Display all suggestions in expandable cards
    if suggestions:
        for suggestion in suggestions:
            with st.expander(
                f"{suggestion['icon']}  {suggestion['title']}"
            ):
                st.markdown(suggestion["text"])
    else:
        st.success(
            "🎉 Your resume looks great for this position! "
            "No major issues detected."
        )
    
    
    # ── RESULT BLOCK 7: DOWNLOAD REPORT ────────────────────────
    
    st.markdown("---")
    st.subheader("📥 Download Your Report")
    
    # Build a text report of everything
    report_text = f"""
AI RESUME ANALYSIS REPORT
{'='*50}
Generated by AI Resume Analyzer

OVERALL MATCH SCORE: {match_results['final_score']}%
TF-IDF Score:        {match_results['tfidf_score']}%
Skill Match Score:   {match_results['skill_score']}%

{'='*50}
CONTACT INFORMATION
{'='*50}
Name:       {entities.get('name', 'Not detected')}
Email:      {entities.get('email', 'Not detected')}
Phone:      {entities.get('phone', 'Not detected')}
LinkedIn:   {entities.get('linkedin', 'Not detected')}
Experience: {str(entities.get('experience_years', 'N/A')) + ' years'}

{'='*50}
SKILLS MATCHED ({len(skill_gaps['matched'])}/{skill_gaps['jd_total']})
{'='*50}
{', '.join(skill_gaps['matched']) if skill_gaps['matched'] else 'None'}

{'='*50}
MISSING SKILLS
{'='*50}
{', '.join(skill_gaps['missing']) if skill_gaps['missing'] else 'None - All skills present!'}

{'='*50}
ALL RESUME SKILLS DETECTED
{'='*50}
{', '.join(entities['skills'])}

{'='*50}
EDUCATION DETECTED
{'='*50}
{chr(10).join(entities['education']) if entities['education'] else 'Not detected'}

{'='*50}
ML MODEL PREDICTION
{'='*50}
Predicted Category: {category_result['predicted_category']}
Confidence:         {category_result['confidence']}%

Top 3 Predictions:
{chr(10).join([f"  {p['category']}: {p['probability']}%" for p in category_result['top3_predictions']])}

{'='*50}
SUGGESTIONS
{'='*50}
{chr(10).join([f"• {s['title']}: {s['text']}" for s in suggestions])}
"""
    
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=report_text,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        help="Download a complete text report of your analysis"
    )
    