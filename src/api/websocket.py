import uuid
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .schemas import OptimizationRequest, OptimizationResponse
from .service import OptimizationService

router = APIRouter()
service = OptimizationService()

@router.websocket("/ws/optimize")
async def websocket_optimize(websocket: WebSocket):
    await websocket.accept()
    run_id = str(uuid.uuid4())
    
    try:
        # Wait for client to send the request payload
        data = await websocket.receive_json()
        
        try:
            request_data = OptimizationRequest(**data)
            sym_request = request_data.to_symbolic_request()
        except ValidationError as e:
            await websocket.send_json({
                "event": "error",
                "run_id": run_id,
                "message": f"Validation error: {e.errors()}"
            })
            await websocket.close()
            return
            
        await websocket.send_json({
            "event": "started",
            "run_id": run_id,
            "message": "Optimization started"
        })
        
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        
        def sync_progress_callback(event_data: dict):
            # Inject run_id into every event
            event_data["run_id"] = run_id
            # Thread-safe queue put
            loop.call_soon_threadsafe(queue.put_nowait, event_data)
            
        # We start the optimization in a background task
        opt_task = asyncio.create_task(
            service.run_pipeline_async(sym_request, progress_callback=sync_progress_callback)
        )
        
        # We loop and consume the queue while the task is running
        while not opt_task.done() or not queue.empty():
            try:
                # Wait for the next event or until the task completes
                # We use a short timeout to periodically check if the task is done
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                # Client disconnected, we should attempt to cancel or just abandon
                # Since GA/PSO are synchronous CPU-bound in a thread, we can't easily cancel them.
                # Just break and let it finish in the void.
                break
                
        if not opt_task.cancelled() and opt_task.done() and websocket.client_state.name == "CONNECTED":
            try:
                result = opt_task.result()
                response = OptimizationResponse.from_result(result)
                await websocket.send_json({
                    "event": "completed",
                    "run_id": run_id,
                    "result": response.model_dump()
                })
            except Exception as e:
                await websocket.send_json({
                    "event": "error",
                    "run_id": run_id,
                    "message": f"Internal optimization error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json({
                "event": "error",
                "run_id": run_id,
                "message": f"Unexpected error: {str(e)}"
            })
        
    if websocket.client_state.name == "CONNECTED":
        await websocket.close()
