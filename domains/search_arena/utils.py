from agent.llm import OPENAI_MODEL

QUESTION_ID = "question_id"
GROUND_TRUTH_KEY = "winner"
MODEL = OPENAI_MODEL

def format_input_dict(row):
    # Extract the inputs for the task from the row
    return {
        "domain": "search_arena",
        "messages_a": row['messages_a'],
        "messages_b": row['messages_b'],
    }
