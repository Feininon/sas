from text_pdf_splitter import split_files
from function_calling import process_directory
import numpy as np

def rank_documents(input_path):
    
    # Get budget from the user
    try:
        budget = float(input("Enter the maximum allowable budget (finance limit): "))
    except ValueError:
        print("Invalid input! Please enter a valid number for the budget.")
        return

    # Split input files into text and PDF components
    processed_files_path = split_files(input_path)

    # Process the directory to extract ESG scores, risks, and finance values
    scores, insights, risks = process_directory(processed_files_path)

    # Convert data into structured document representation
    num_documents = scores.shape[0]
    documents = []
    for i in range(num_documents):
        documents.append({
            "document_id": i,
            "esg_score": scores[i, 0],  # Assuming ESG score is in column 0
            "risk": risks[i, 0],       # Assuming risk is in column 0
            "finance": insights[i, 0]  # Assuming finance cost is in column 0
        })

    # Filter and rank documents based on selection criteria
    ranked_documents = sorted(
        [doc for doc in documents if doc["finance"] <= budget],
        key=lambda x: (-x["esg_score"], x["risk"])  # Higher ESG score, lower risk
    )

    # Display the ranked documents
    if ranked_documents:
        print("\nRanked Documents (Within Budget):")
        for rank, doc in enumerate(ranked_documents, start=1):
            print(f"Rank {rank}:")
            print(f"  Document ID: {doc['document_id']}")
            print(f"  ESG Score: {doc['esg_score']}")
            print(f"  Risk: {doc['risk']}")
            print(f"  Finance Cost: {doc['finance']}")
    else:
        print("No documents meet the budget constraint.")

    return ranked_documents

# Example usage
# input_path = "/path/to/input/files"
# ranked_documents = rank_documents(input_path)