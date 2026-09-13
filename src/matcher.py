"""
matcher.py

Resume <-> Job Description matching.

Two signals are combined into the final "Resume Match Score":

1. Text similarity  - TF-IDF vectors of the full resume/JD text compared
                       with cosine similarity. This captures overall
                       vocabulary/content overlap but is naturally low
                       (often 10-30%) even for good matches, since resumes
                       contain many words (dates, company names, formatting
                       artifacts) that never appear in a job description.

2. Skill coverage   - the fraction of skills required by the job
                       description that are actually present in the resume
                       (from src.skill_extractor). This is closer to how
                       most commercial "ATS score" tools actually work,
                       since they focus on keyword/skill matching rather
                       than whole-document similarity.

The final score is a weighted blend of the two, which produces numbers in
a range closer to what people expect from resume-matching tools while
remaining fully explainable (no hidden inflation).

This is still a similarity score, not a probability of getting hired.
"""

from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.skill_extractor import compare_skills

# Weight given to skill-keyword coverage vs. raw text similarity when
# computing the final blended match score.
SKILL_WEIGHT = 0.6
TEXT_WEIGHT = 0.4


def compute_text_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute raw TF-IDF + cosine similarity between resume and JD text.

    A fresh TF-IDF vectorizer is fit on just these two documents so the
    score reflects how similar the resume's vocabulary/content is to the
    specific job description provided.

    Args:
        resume_text: Raw text extracted from the candidate's resume.
        jd_text: Raw text of the job description.

    Returns:
        Similarity score as a percentage (0.0 - 100.0), rounded to 2 decimals.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    if not cleaned_resume or not cleaned_jd:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute the blended Resume Match Score.

    score = TEXT_WEIGHT * text_similarity + SKILL_WEIGHT * skill_coverage

    If the job description has no recognizable skills at all (so skill
    coverage can't be computed), the score falls back to pure text
    similarity so the result is never artificially zeroed out.

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.

    Returns:
        Match score as a percentage (0.0 - 100.0), rounded to 2 decimals.
    """
    text_score = compute_text_similarity(resume_text, jd_text)
    skills = compare_skills(resume_text, jd_text)

    if skills["jd_skills"]:
        blended = (TEXT_WEIGHT * text_score) + (SKILL_WEIGHT * skills["match_percentage"])
    else:
        blended = text_score

    return round(blended, 2)


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

    if score >= 75:
        label = "Strong Match"
    elif score >= 50:
        label = "Good Match"
    elif score >= 30:
        label = "Moderate Match"
    else:
        label = "Weak Match"

    return score, label
