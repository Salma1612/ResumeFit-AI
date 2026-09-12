"""
matcher.py

Resume <-> Job Description matching using TF-IDF vectorization and
cosine similarity. This produces a similarity-based "Resume Match Score"
(NOT a probability of getting hired -- see README limitations section).
"""

from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute a similarity-based match score between a resume and a job description.

    A fresh TF-IDF vectorizer is fit on just these two documents so the
    score reflects how similar the resume's vocabulary/content is to the
    specific job description provided (a local, query-time comparison,
    distinct from the globally-trained classifier used for category
    prediction).

    Args:
        resume_text: Raw text extracted from the candidate's resume.
        jd_text: Raw text of the job description.

    Returns:
        Match score as a percentage (0.0 - 100.0), rounded to 2 decimals.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    if not cleaned_resume or not cleaned_jd:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score = round(float(similarity) * 100, 2)
    return score


def match_summary(resume_text: str, jd_text: str) -> Tuple[float, str]:
    """
    Compute the match score and a short human-readable interpretation.

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.

    Returns:
        Tuple of (score, interpretation_label)
    """
    score = compute_match_score(resume_text, jd_text)

    if score >= 70:
        label = "Strong Match"
    elif score >= 45:
        label = "Moderate Match"
    elif score >= 20:
        label = "Weak Match"
    else:
        label = "Very Low Match"

    return score, label
