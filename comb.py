from pdfplumber import open as open_pdf
from textModel1 import score_project_impact as model1
from textModel2 import score_project_impact as model2
from textModel3 import extract_insights_from_text as model3
from textModel4 import predict_risks_from_text as model4
from textModel5 import classify_text as model5
from textModel6 import classify_text as model6
import os
import numpy as np


def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text content.
    """
    text_content = ""
    with open_pdf(pdf_path) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text()
    return text_content


def extract_text_from_txt(txt_path):
    """
    Extract text content from a TXT file.
    
    Args:
        txt_path (str): Path to the text file.
        
    Returns:
        str: Extracted text content.
    """
    with open(txt_path, "r") as file:
        return file.read()


def process_directory(directory_path):
    """
    Process all text and PDF files in the given directory and apply multiple models to extract insights, score projects, and predict risks.
    
    Args:
        directory_path (str): Path to the directory containing text and PDF files.
        
    Returns:
        tuple: Matrices for project scores, extracted insights, and predicted risks.
    """
    ProjectScores = []  # To store project scores from different models
    Insights = []       # To store extracted insights
    RiskPredictions = []  # To store predicted risks
    
    # Loop through all text and PDF files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        text_content = ""
        
        # Check file type and extract content
        if file_name.endswith(".txt"):
            text_content = extract_text_from_txt(file_path)
        elif file_name.endswith(".pdf"):
            text_content = extract_text_from_pdf(file_path)
        else:
            continue  # Skip unsupported file formats
        
        # Get predictions/scores for the current file from all models
        scores = []
        insights = []
        risks = []
        
        # Score project impact
        scores.append(model1(text_content))
        scores.append(model2(text_content))
        
        # Extract insights
        insights.append(model3(text_content))
        
        # Predict risks
        risks.append(model4(text_content))
        
        # Classify text (e.g., categories, tags, or other attributes)
        risks.append(model5(text_content))
        risks.append(model6(text_content))
        
        # Append results to the respective lists
        ProjectScores.append(scores)
        Insights.append(insights)
        RiskPredictions.append(risks)
    
    # Convert results into NumPy arrays (matrix form)
    score_matrix = np.array(ProjectScores)
    insight_matrix = np.array(Insights)
    risk_matrix = np.array(RiskPredictions)
    
    # Return matrices of results
    return score_matrix, insight_matrix, risk_matrix