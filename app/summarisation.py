from transformers import pipeline
import logging
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pipeline(model_name: str):
    """Initialize the summarisation pipeline"""
    logger.info(f"Initializing summarisation pipeline with model: {model_name}")
    try:
        summarisation_pipeline = pipeline("text-generation", model=model_name)
        logger.info("Pipeline initialized successfully")
        return summarisation_pipeline
    except Exception as e:
        logger.error(f"Error initializing pipeline: {str(e)}")
        raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

def parse_json(generated_text: str) -> dict:
    """Parse JSON output."""
    # Extract JSON using regex
    json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
    if not json_match:
        raise ValueError("Model output did not contain valid JSON.")
    json_text = json_match.group(0)

    try:
        output = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {json_text}. Error: {str(e)}")
    return output

def generate_summary(summarisation_pipeline, review_text: str) -> dict:
    """Generate a summary for an input string using the summarisation pipeline."""
    # Construct the prompt
    prompt = (
        "Your task is to extract specific information from the given review and present it in valid JSON format.\n"
        "Extract the following information:\n"
        "1. title: A concise title of the review (string).\n"
        "2. summary: A short summary of the review in 2-3 sentences (string).\n"
        "3. grade: A grade for the movie between 0 and 5 (integer).\n"
        "The output must strictly follow this JSON format:\n"
        '{\n'
        '  "title": "Title of the summary",\n'
        '  "summary": "This is the summary of the review.",\n'
        '  "grade": 4\n'
        '}\n\n'
        f"Review: {review_text}\n\n"
        "Output only the JSON object, with no extra text or comments."
    )

    # Generate text
    results = summarisation_pipeline(prompt, max_length=500, temperature=0.2, top_p=0.9, do_sample=True)
    generated_text = results[0]["generated_text"]

    # Parse the JSON
    try:
        output = parse_json(generated_text)
    except ValueError as e:
        raise ValueError(f"Model output error: {str(e)}")

    # Validate the parsed output
    required_fields = ["title", "summary", "grade"]
    for field in required_fields:
        if field not in output:
            raise ValueError(f"Missing required field: {field}")

    # Validate the grade
    if not isinstance(output["grade"], int) or not (0 <= output["grade"] <= 5):
        raise ValueError("Grade must be an integer between 0 and 5.")

    return output
   