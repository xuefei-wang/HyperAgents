from agent.llm import OPENAI_MODEL

QUESTION_ID = "question_id"
GROUND_TRUTH_KEY = "outcome"
MODEL = OPENAI_MODEL

def format_input_dict(row):
    # Extract the inputs for the task from the row
    return {
        "domain": "paper_review",
        "paper_text": row['paper_text'],
    }
