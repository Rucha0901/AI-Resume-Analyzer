# test_similarity.py
# Test the similarity scorer with different resume/JD combinations

from pdf_extractor import extract_text_from_pdf
from similarity_matcher import analyze_match, print_match_results

# ── TEST 1: Strong Match ──────────────────────────────
# Use a data science resume with a data science JD
# Expected score: HIGH (70-90%)

resume_text_1 = """
John Doe
Email: john@gmail.com | Phone: 9876543210

EXPERIENCE
Data Scientist at Google, 2020-2023
- Built machine learning models using Python, TensorFlow, scikit-learn
- Data analysis and visualization using Pandas, Matplotlib
- SQL queries on large datasets in BigQuery
- Deployed models using Docker and AWS

SKILLS
Python, SQL, Machine Learning, Deep Learning, TensorFlow, 
Pandas, NumPy, Scikit-learn, Docker, AWS, Data Analysis

EDUCATION
M.Tech Data Science, IIT Bombay, 2018-2020
"""

jd_text_1 = """
Data Scientist — We are looking for an experienced Data Scientist.

Requirements:
- Strong Python programming skills
- Experience with Machine Learning and Deep Learning
- Proficiency in TensorFlow or PyTorch
- SQL and data analysis skills
- Experience with cloud platforms like AWS or GCP
- Familiarity with Docker and deployment pipelines
- Knowledge of Pandas, NumPy, Scikit-learn

Nice to have:
- Experience with big data tools
- Strong communication and presentation skills
"""

# ── TEST 2: Weak Match ────────────────────────────────
# Use a web developer resume with a data science JD
# Expected score: LOW (20-40%)

resume_text_2 = """
Jane Smith
Email: jane@gmail.com

EXPERIENCE
Frontend Developer at Startup, 2021-2023
- Built web applications using React and JavaScript
- HTML, CSS, Bootstrap for UI design
- Node.js and Express for backend APIs
- MongoDB database management
- Git version control

SKILLS
JavaScript, React, HTML, CSS, Node.js, MongoDB, Git, Bootstrap

EDUCATION
B.Tech Computer Science, 2017-2021
"""

# ── TEST 3: Medium Match ──────────────────────────────
# Resume has SOME but not all required skills
# Expected score: MEDIUM (45-65%)

resume_text_3 = """
Raj Patel
Email: raj@gmail.com

EXPERIENCE
Software Engineer at Infosys, 2019-2023
- Python development for automation scripts
- SQL database management and query optimization
- Basic data analysis using Pandas
- REST API development using Flask

SKILLS
Python, SQL, Pandas, Flask, Git, Linux, REST API

EDUCATION
B.Tech Information Technology, 2015-2019
"""

# ── RUN ALL TESTS ─────────────────────────────────────

test_cases = [
    {
        "label"  : "TEST 1: Data Science Resume vs Data Science JD (EXPECT HIGH)",
        "resume" : resume_text_1,
        "jd"     : jd_text_1
    },
    {
        "label"  : "TEST 2: Web Dev Resume vs Data Science JD (EXPECT LOW)",
        "resume" : resume_text_2,
        "jd"     : jd_text_1
    },
    {
        "label"  : "TEST 3: Backend Dev Resume vs Data Science JD (EXPECT MEDIUM)",
        "resume" : resume_text_3,
        "jd"     : jd_text_1
    },
]

for test in test_cases:
    print(f"\n\n{'#'*55}")
    print(f"  {test['label']}")
    print(f"{'#'*55}")
    
    results = analyze_match(test["resume"], test["jd"])
    print_match_results(results)
    
    input("\nPress Enter to run next test...")


# ── TEST WITH REAL PDFs ───────────────────────────────
print("\n\nNow testing with your actual PDF resumes...")

real_jd = """
Software Engineer - Python Backend
Requirements: Python, Django or Flask, REST APIs, SQL, PostgreSQL,
Docker, AWS, Git, Agile, problem solving, teamwork, communication
Experience: 2-4 years in backend development
"""

resume_pdf_path = "sample_resumes/resume1.pdf"

from pdf_extractor import extract_text_from_pdf
real_resume_text = extract_text_from_pdf(resume_pdf_path)

real_results = analyze_match(real_resume_text, real_jd)
print_match_results(real_results)