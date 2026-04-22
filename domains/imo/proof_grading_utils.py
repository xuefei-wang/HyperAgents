from agent.llm import OPENAI_MODEL

QUESTION_ID = "Problem ID"
GROUND_TRUTH_KEY = None  # Ground truth does not exist
MODEL = OPENAI_MODEL

def format_input_dict(row):
    # Extract the inputs for the task from the row
    return {
        "domain": "imo_proof_grading",
        "problem": row['Problem'],
        "solution": row['Solution'],
        "grading_guidelines": row['Grading guidelines'],
        "student_answer": row['Response'],
    }
