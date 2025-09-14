from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def hello():
    return {"msg": "hello from 3.10.8 on amd64"}
