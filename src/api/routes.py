from fastapi import APIRouter, HTTPException
from .schemas import OptimizationRequest, HealthResponse, OptimizationResponse, NaturalLanguageQueryRequest, NaturalLanguageQueryResponse
from .service import OptimizationService

router = APIRouter()
service = OptimizationService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Lightweight health check endpoint."""
    return HealthResponse(status="ok")

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    """
    Synchronous POST endpoint (executed asynchronously via thread).
    Validates the request, executes the full symbolic pipeline, and returns the result.
    """
    try:
        sym_request = request.to_symbolic_request()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request parameters: {str(e)}")
        
    try:
        result = await service.run_pipeline_async(sym_request)
        return OptimizationResponse.from_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal optimization error: {str(e)}")

@router.post("/optimize/query", response_model=NaturalLanguageQueryResponse)
async def optimize_query(request: NaturalLanguageQueryRequest):
    """
    End-to-end endpoint: takes natural language, parses it, routes to solver, and returns report.
    """
    from src.orchestrator.service import NeuroSymbolicOrchestrator
    try:
        orchestrator = NeuroSymbolicOrchestrator()
        report = orchestrator.process_query(request.query)
        return NaturalLanguageQueryResponse(report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
