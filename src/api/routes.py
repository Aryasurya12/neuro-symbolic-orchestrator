from fastapi import APIRouter, HTTPException
from .schemas import OptimizationRequest, HealthResponse, OptimizationResponse
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
