from fastapi import FastAPI, HTTPException
from app.summarisation import init_pipeline, generate_summary
from app.models import MovieReview, MovieSummary
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting API and initializing the summarisation pipeline")
summarisation_pipeline = init_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

app = FastAPI(
    title="Movie Review Summarisation API",
    version="0.1.0"
)

@app.post("/movie_summary", response_model=MovieSummary)
def movie_summary(movie_review: MovieReview):
    """API endpoint to summarise a movie review."""
    logger.info(f"Received request: {movie_review}")
    try:
        model_output = generate_summary(summarisation_pipeline, movie_review.review)
        return MovieSummary(
            title=model_output["title"],
            summary=model_output["summary"],
            grade=model_output["grade"],
            reviewer=movie_review.reviewer
        )
    except ValueError as e:
        logger.error(f"Invalid model output: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Model output error: {str(e)}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
