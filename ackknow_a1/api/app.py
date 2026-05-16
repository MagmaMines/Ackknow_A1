from fastapi import FastAPI
from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.engine import AckknowA1
from ackknow_a1.types import AckRequest, AckResponse

settings = AckknowSettings()
engine = AckknowA1(settings)
app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get('/health')
async def health() -> dict[str, str]:
    return {"status": "ok", "name": settings.app_name, "version": settings.app_version}


@app.post('/v1/chat', response_model=AckResponse)
async def chat(req: AckRequest) -> AckResponse:
    return await engine.run(req)
