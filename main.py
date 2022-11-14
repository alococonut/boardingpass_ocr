import time
import uuid
import uvicorn

from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.context import correlation_id
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse

import routers.ocr_router
import dependencies
from commons.logger import logger as logging, init_uvicorn_logger

correlation_id.set(uuid.uuid4().hex)
init_uvicorn_logger()
app: FastAPI = FastAPI(title="Income BoardingPass OCR",
                       description=dependencies.description,
                       version="0.0.1",
                       license_info={
                           "name": "Apache 2.0",
                           "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
                       },
                       x_logo={
                           "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
                       },
                       root_path="",
                       servers=[
                           {"url": "/", "description": "dev environment"}
                       ]
                       )
app.include_router(routers.ocr_router.router)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exception: Exception) -> JSONResponse:
    return await http_exception_handler(
        request,
        HTTPException(
            500,
            'Internal server error',
            headers={
                'X-Request-ID': correlation_id.get() or "",
                'Access-Control-Expose-Headers': 'x-request-id'
            }
        ))


@app.middleware("http")
async def middleware(request: Request, call_next):
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.on_event("startup")
async def startup_event():
    app.add_middleware(CorrelationIdMiddleware)
    logging.info("app startup")


@app.on_event("shutdown")
async def on_shutdown():
    logging.info("app shutdown")


@app.get("/")
def root():
    return 'Income boarding pass OCR'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
