# skills_database.py
# This is your master list of skills to look for in resumes
# The more complete this list, the better your analyzer works
# This is YOUR custom contribution to the project

SKILLS_DATABASE = {
    
    # ─────────────────────────────────────────
    # PROGRAMMING LANGUAGES
    # ─────────────────────────────────────────
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "r",
        "matlab", "perl", "bash", "shell", "vba", "dart", "lua",
        "assembly", "cobol", "fortran", "haskell", "elixir"
    ],
    
    # ─────────────────────────────────────────
    # WEB DEVELOPMENT
    # ─────────────────────────────────────────
    "web_development": [
        "html", "css", "react", "angular", "vue", "nodejs", "express",
        "django", "flask", "fastapi", "spring", "laravel", "rails",
        "jquery", "bootstrap", "tailwind", "sass", "webpack", "redux",
        "graphql", "restapi", "rest", "soap", "ajax", "jsp", "asp"
    ],
    
    # ─────────────────────────────────────────
    # DATA SCIENCE & MACHINE LEARNING
    # ─────────────────────────────────────────
    "data_science_ml": [
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "nlp", "computer vision",
        "data science", "data analysis", "data mining",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "opencv", "huggingface", "transformers", "bert", "gpt",
        "regression", "classification", "clustering", "random forest",
        "xgboost", "lightgbm", "svm", "cnn", "rnn", "lstm",
        "reinforcement learning", "transfer learning", "feature engineering"
    ],
    
    # ─────────────────────────────────────────
    # DATABASES
    # ─────────────────────────────────────────
    "databases": [
        "sql", "mysql", "postgresql", "sqlite", "oracle", "mongodb",
        "redis", "cassandra", "elasticsearch", "dynamodb", "firebase",
        "mariadb", "neo4j", "hbase", "influxdb", "nosql",
        "database design", "query optimization", "stored procedures"
    ],
    
    # ─────────────────────────────────────────
    # CLOUD & DEVOPS
    # ─────────────────────────────────────────
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "jenkins", "git", "github", "gitlab", "bitbucket", "ci/cd",
        "terraform", "ansible", "linux", "unix", "nginx", "apache",
        "microservices", "serverless", "lambda", "ec2", "s3",
        "devops", "agile", "scrum", "jira", "confluence"
    ],
    
    # ─────────────────────────────────────────
    # DATA ENGINEERING & BIG DATA
    # ─────────────────────────────────────────
    "data_engineering": [
        "hadoop", "spark", "kafka", "airflow", "hive", "pig",
        "data pipeline", "etl", "data warehouse", "data lake",
        "tableau", "power bi", "looker", "qlik", "superset",
        "bigquery", "redshift", "snowflake", "databricks"
    ],
    
    # ─────────────────────────────────────────
    # MOBILE DEVELOPMENT
    # ─────────────────────────────────────────
    "mobile": [
        "android", "ios", "react native", "flutter", "xamarin",
        "swift", "objective-c", "android studio", "xcode"
    ],
    
    # ─────────────────────────────────────────
    # SOFT SKILLS
    # ─────────────────────────────────────────
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "critical thinking", "project management", "time management",
        "collaboration", "adaptability", "creativity", "analytical",
        "presentation", "negotiation", "mentoring", "research",
        "attention to detail", "multitasking", "decision making",
        "conflict resolution", "strategic planning", "innovation"
    ],
    
    # ─────────────────────────────────────────
    # BUSINESS & MANAGEMENT
    # ─────────────────────────────────────────
    "business": [
        "product management", "business analysis", "stakeholder management",
        "budget management", "risk management", "change management",
        "digital marketing", "seo", "content marketing", "crm",
        "salesforce", "erp", "sap", "microsoft office", "excel",
        "powerpoint", "word", "financial modeling", "forecasting"
    ],
    
    # ─────────────────────────────────────────
    # CYBERSECURITY
    # ─────────────────────────────────────────
    "security": [
        "cybersecurity", "network security", "penetration testing",
        "ethical hacking", "firewalls", "encryption", "ssl",
        "vulnerability assessment", "siem", "soc", "compliance",
        "iso 27001", "gdpr", "owasp"
    ]
}


def get_all_skills():
    """
    Flattens the nested dictionary into one big flat list of all skills.
    This is used when checking if a word appears in ANY skill category.
    
    Example output:
    ["python", "java", "html", "machine learning", "sql", ...]
    """
    all_skills = []
    for category, skills_list in SKILLS_DATABASE.items():
        all_skills.extend(skills_list)
    return all_skills


def get_skills_by_category():
    """Returns the full nested dictionary - used when you want category-wise breakdown"""
    return SKILLS_DATABASE
