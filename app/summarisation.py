from transformers import pipeline
import logging
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

def parse_summary(generated_text: str) -> str:
    """Parse the summary from the model's output."""
    # Define a regex to extract text following <|assistant|>
    summary_pattern = r"<\|assistant\|>\s*(.*)"
    match = re.search(summary_pattern, generated_text, re.DOTALL)

    if match:
        # Return the summary text
        return match.group(1).strip()
    else:
        # If no match, log and raise an error
        logger.error("Failed to parse the summary from the model output.")
        raise ValueError("Failed to parse summary from model output.")

def generate_summary(summarisation_pipeline, review_text: str) -> dict:
    """Generate a summary for a given movie review using the summarisation pipeline."""
    # Define the system-level instruction
    system_prompt = "You are a helpful assistant tasked with writing concise summaries of movie reviews."

    # Define the user-specific prompt
    prompt = f"Please summarise the following movie review in exactly 2-3 sentences:\n\nReview: {review_text} "

    # Combine into the formatted prompt
    formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

    try:
        # Generate response
        logger.info("Sending prompt to the model...")

        results = summarisation_pipeline(
            formatted_prompt,
            do_sample=True,
            top_k=50,
            top_p=0.6,
            temperature=0.1,
            num_return_sequences=1,
            repetition_penalty=1.2,
            max_new_tokens=150,
        )
        generated_text = results[0]["generated_text"].strip()

        # Log the raw model output
        logger.info(f"Raw model output:\n{generated_text}")

        # Parse the relevant part of the output
        summary = parse_summary(generated_text)

        # Return the summary with hardcoded fields
        return {
            "title": "Review Title",  # default
            "summary": summary,
            "grade": 4,  # default
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError("Unexpected error during summary generation.")

   