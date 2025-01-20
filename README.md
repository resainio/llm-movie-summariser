# Movie Summariser API
A FastAPI-based service for movie review summaries using the [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0). 
It's purpose is to generate the following fields for a given review:
1. A **Title** for the review.
2. A short **Summary** for the review.
3. A **Grade** between 0 and 5 based on the review content.

## Requirements
Make sure you have **Docker** installed to build and run the API.

## Run the API
1. Build the Docker image:
    ```docker build -t movie-summary-api .```

2. Run the Docker container:
   ```docker run -d --name moviecontainer -p 8000:8000 movie-summary-api```

3. The API will be available at:
    Swagger UI: http://localhost:8000/docs

## Test the API
You can test the API with the following review example:

```{
  "review": "I recently watched Alex Garlands film Ex Machina, who is also known for films such as Sunshine and Annihilation. The premise of the film is a young programmer tasked with determining whether an advanced AI possesses true consciousness, and it explores the interesting philosophical dimensions of artificial intelligence and consciousness. In general Garland is able to build an interesting atmosphere and the cinematography is excellent. However, I found that the movie is a bit too slow in the beginning and the characters are only explored at surface level, which personally left me wanting for a bit more. The issues the movie tackles are also quite simplified, possibly to make the film a bit more digestible, but I personally felt that a few less clichés would have improved the film. Reviewed by Jr Robot 123",
  "reviewer": "Jr Robot 123"
}```
