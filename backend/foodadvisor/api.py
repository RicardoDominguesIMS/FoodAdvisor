from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .business_service import LLMWorkflowError, RecipeBusinessService
from .rag_repository import RAGUnavailableError

app = FastAPI(title="FoodAdvisor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

business_service = RecipeBusinessService()


class RecipeRequest(BaseModel):
    recipe: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/rag/search")
def search_rag(payload: RecipeRequest) -> dict:
    try:
        return business_service.retrieve_recipe_context(payload.recipe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/rag/rebuild")
def rebuild_rag() -> dict:
    try:
        result = business_service.rebuild_rag_index()
    except RAGUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"status": "ok", **result}


@app.post("/api/analyze")
def analyze_recipe(payload: RecipeRequest) -> dict:
    recipe = payload.recipe.strip()
    if not recipe:
        raise HTTPException(status_code=400, detail="A recipe name is required.")

    try:
        result = business_service.analyze_recipe(recipe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMWorkflowError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "detail": str(exc),
                "workflow_log": exc.workflow_log,
            },
        )

    return result.to_dict()
