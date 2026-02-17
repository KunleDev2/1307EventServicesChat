from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_answer

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(question: Question):
    answer = generate_answer(question.question)
    return {"answer": answer}