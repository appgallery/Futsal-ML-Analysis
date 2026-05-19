import uvicorn
from fastapi import FastAPI
from api.routes.predict import lifespan, routes

app = FastAPI(title="Futsal Match Prediction API", lifespan=lifespan)
app.include_router(routes)

def main():
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

if __name__ == "__main__":
    main()
