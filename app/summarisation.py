from transformers import pipeline
import logging
import json

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

def generate_summary(summarisation_pipeline, review_text: str) -> dict:
    """Generate a summary for an input string using the summarisation pipeline"""
    logger.info("Generating title, summary, and grade for the given review")
    try:
        prompt = (
            "Analyse the following review and return a valid JSON object with these fields:\n"
            '{"title": "string",    // A concise title for the review\n'
            '"summary": "string",  // A short summary in 2-3 sentences\n'
            '"grade": integer}     // A grade between 0 and 5\n'
            "Only output valid JSON, with no extra text or comments.\n"
            f"Review: {review_text}"
        )

        results = summarisation_pipeline(prompt, max_length=300, temperature=0.7, top_p=0.90, do_sample=True)
        generated_text = results[0]['generated_text']

        logger.info(f"Generated text: {generated_text}")

        parsed_output = json.loads(generated_text)
        return parsed_output

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {generated_text}. Error: {str(e)}")
        raise ValueError("Invalid JSON output from the model.")
    except Exception as e:
        logger.error(f"Error during summary generation: {str(e)}")
        raise e
