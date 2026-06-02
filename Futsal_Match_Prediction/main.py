import os
import subprocess
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from api.routes.predict import lifespan, routes

app = FastAPI(title="Futsal Match Prediction API", lifespan=lifespan)
app.include_router(routes)

def start_docker_compose():
    if not os.environ.get("IN_DOCKER"):
        print("Starting Docker Compose services (excluding fastapi-app to prevent port collision)...")
        try:
            # We scale fastapi-app to 0 so the Dockerized version doesn't conflict with the local instance
            subprocess.run(["docker-compose", "up", "-d", "--scale", "fastapi-app=0"], check=True)
        except Exception as e:
            print(f"Failed to start docker-compose automatically: {e}")

def main():
    start_docker_compose()
    host = "0.0.0.0" if os.environ.get("IN_DOCKER") else "localhost"
    uvicorn.run("main:app", host=host, port=8000, reload=True)

if __name__ == "__main__":
    main()
