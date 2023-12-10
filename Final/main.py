from fastapi import FastAPI
from wordle_bot import guess_word, initialize, QLearner, Grader, Interpreter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


initialized = False 
interpreter = None
qlearner = None
grader = None


def initialize_once():
    global interpreter, qlearner, grader, initialized
    # print("initialized: ", initialized)
    if not initialized:
        print("Initializing")
        interpreter, qlearner, grader = initialize()
        initialized = True

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/guess/{word}")
def read_item(word: str):
    global interpreter, qlearner, grader, initialized
    initialize_once()
    guesses, interpreter, qlearner, grader = guess_word(word, interpreter, qlearner, grader)
    return {"guesses": guesses, "target": word}

# @app.get("/initialize")
# def initialize():
#     global interpreter, qlearner, grader, initialized
#     initialize_once()
#     return {"initialized": initialized}

if __name__ == "__main__":
    initialize_once()
    uvicorn.run(app, host="127.0.0.1", port=8000)