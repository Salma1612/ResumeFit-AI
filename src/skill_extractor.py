"""
skill_extractor.py

A practical technical-skills dictionary and simple, dependency-free
keyword-based skill extraction used for the Skill Gap Analysis feature.

Matching is done with word-boundary-aware regex over the lowercased text,
so multi-word skills like "machine learning" or "power bi" are matched
correctly and short skills like "r" or "go" don't false-positive on
substrings of other words.
"""

import re
from typing import Dict, List, Set

# A curated, practical technical-skills dictionary grouped by domain.
# Extend this dictionary as needed for other domains (the app works with
# any category present in Resume.csv, but skill matching is strongest
# for tech / data roles).
SKILLS_DB: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "javascript", "typescript", "r",
        "go", "golang", "ruby", "php", "swift", "kotlin", "scala", "rust",
        "matlab", "perl", "sql", "bash", "shell scripting",
    ],
    "Web Development": [
        "html", "css", "react", "angular", "vue", "node.js", "nodejs",
        "django", "flask", "fastapi", "spring boot", "asp.net", "next.js",
        "rest api", "graphql", "bootstrap", "tailwind css", "jquery",
    ],
    "Data Science & ML": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "data analysis", "data visualization",
        "statistics", "pandas", "numpy", "scikit-learn", "tensorflow",
        "keras", "pytorch", "xgboost", "opencv", "feature engineering",
        "predictive modeling", "time series analysis", "a/b testing",
    ],
    "Data Engineering & Big Data": [
        "spark", "hadoop", "kafka", "airflow", "etl", "data pipeline",
        "data warehousing", "hive", "databricks", "snowflake",
        "big data", "dbt",
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "oracle", "sqlite",
        "microsoft sql server", "redis", "cassandra", "dynamodb",
        "elasticsearch",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "google cloud platform", "gcp", "docker",
        "kubernetes", "terraform", "jenkins", "ci/cd", "ansible",
        "linux", "git", "github", "gitlab", "devops", "microservices",
    ],
    "BI & Visualization Tools": [
        "power bi", "tableau", "excel", "looker", "qlikview",
        "google analytics", "google data studio",
    ],
    "Project & Soft Skills": [
        "agile", "scrum", "jira", "project management", "communication",
        "leadership", "problem solving", "teamwork", "stakeholder management",
        "time management", "critical thinking",
    ],
    "Other Technical": [
        "api development", "unit testing", "object oriented programming",
        "data structures", "algorithms", "system design", "automation",
        "cybersecurity", "networking", "blockchain",
    ],
}


def _flatten_skills() -> List[str]:
    """Return a flat, de-duplicated list of all known skills."""
    seen: Set[str] = set()
    flat: List[str] = []
    for skills in SKILLS_DB.values():
        for skill in skills:
            key = skill.lower().strip()
            if key not in seen:
                seen.add(key)
                flat.append(key)
    # Longer phrases first so multi-word skills are matched before
    # any shorter skill that might be a substring of them.
    flat.sort(key=len, reverse=True)
    return flat


ALL_SKILLS: List[str] = _flatten_skills()


def _build_pattern(skill: str) -> re.Pattern:
    """Build a case-insensitive, word-boundary-safe regex for a skill phrase."""
    escaped = re.escape(skill)
    # Allow '.', '+', '#' inside tech terms (e.g. c++, node.js) by using
    # lookaround boundaries instead of strict \b which can fail on symbols.
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"
    return re.compile(pattern, re.IGNORECASE)


_SKILL_PATTERNS = {skill: _build_pattern(skill) for skill in ALL_SKILLS}


def extract_skills(text: str) -> List[str]:
    """
    Extract known technical skills mentioned in the given text.

    Args:
        text: Raw or cleaned text (resume text or job description text).

    Returns:
        Sorted list of unique matched skills (lowercase, as defined in SKILLS_DB).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    lowered = text.lower()
    found = set()
    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(lowered):
            found.add(skill)

    return sorted(found)


def compare_skills(resume_text: str, jd_text: str) -> Dict[str, object]:
    """
    Compare skills present in a resume against skills required by a job description.

    Args:
        resume_text: Text extracted from the candidate's resume.
        jd_text: Text of the target job description.

    Returns:
        A dictionary with:
            - "resume_skills": list of skills found in the resume
            - "jd_skills": list of skills found in the job description
            - "matching_skills": skills present in both
            - "missing_skills": skills required by JD but absent from resume
            - "match_percentage": float percentage of JD skills covered by resume
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    matching_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    if jd_skills:
        match_percentage = round((len(matching_skills) / len(jd_skills)) * 100, 2)
    else:
        match_percentage = 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "match_percentage": match_percentage,
    }
