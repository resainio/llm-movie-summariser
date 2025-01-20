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

def extract_output(generated_text: str) -> dict:
    """
    Extract the title, grade, and summary from the generated text.
    """
    try:
        # Split the generated text by the assistant token
        assistant_split = generated_text.split("<|assistant|>")
        if len(assistant_split) == 2:
            response_text = assistant_split[1].strip()
        else:
            raise ValueError("Failed to parse the output from the model response.")

        # Split the response text into lines
        lines = response_text.splitlines()

        # Extract the title
        title = lines[0].split("Title:")[-1].strip()

        # Extract the grade
        grade_line = lines[1]
        grade_match = re.search(r"Grade:\s*(\d)", grade_line)
        if grade_match:
            grade = int(grade_match.group(1))
        else:
            raise ValueError("Grade not found in the output.")

        # Extract the summary 
        summary = ' '.join(lines[2:]).split("Summary:")[-1].strip()

        return {
            "title": title,
            "grade": grade,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error extracting title, grade, or summary: {str(e)}")
        raise ValueError("Failed to extract title, grade, or summary from generated text.")

def generate_summary(summarisation_pipeline, review_text: str) -> dict:
    """Generate a summary for a given movie review using the summarisation pipeline."""
    # Define the system-level instruction
    system_prompt = "You are a helpful assistant tasked with writing concise summaries of movie reviews."
    
    # Include few-shot prompting
    few_shot_prompt = """
    Review: This movie was an absolute masterpiece! The acting, directing, and story were all phenomenal. I was on the edge of my seat the entire time. The pacing was perfect, and the music elevated the emotional scenes. I would recommend it to anyone who loves cinema at its finest.
    Title: Cinematic Brilliance
    Grade: 5
    Summary: A thrilling masterpiece with exceptional acting, directing, and storytelling that kept the reviewer on the edge of their seat. The pacing and music added to the emotional impact.

    Review: This movie was terrible. The acting was bad, the story was boring, and the special effects were awful. It felt like the film had no direction, and the characters were completely one-dimensional. The dialogue was cringe-worthy, and the ending made no sense. I cannot recommend this to anyone.
    Title: A Painful Watch
    Grade: 1
    Summary: A poorly executed film with bad acting, a dull story, and subpar special effects. The reviewer found the dialogue cringe-worthy and the ending nonsensical.
    """
    
    # Define the user-specific prompt
    prompt = (
        f"Please extract a concise title for the review that captures the essence of the review. "
        f"Please summarise the following movie review in exactly 2-3 sentences. "
        f"Also, assign a grade for the movie between 0 and 5 based on the review content. "
        f"Review: {review_text}"
    )
    # Combine into the formatted prompt
    formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{few_shot_prompt} {prompt}</s>\n<|assistant|>"

    try:
        # Generate response
        logger.info("Sending prompt to the model...")
        results = summarisation_pipeline(
            formatted_prompt,
            do_sample=True,
            top_k=50,
            top_p=0.6,
            temperature=0.2,
            num_return_sequences=1,
            repetition_penalty=1.2,
            max_new_tokens=150,
        )
        generated_text = results[0]["generated_text"].strip()

        # Log the raw model output
        logger.info(f"Raw model output:\n{generated_text}")

        # Parse the relevant parts of the output
        extracted_data = extract_output(generated_text)

        return {
            "title": extracted_data["title"],
            "summary": extracted_data["summary"],
            "grade": extracted_data["grade"],
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError("Unexpected error during summary generation.")
        

   