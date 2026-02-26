import asyncio
import gzip
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware import cors, trustedhost
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from depth2metric.common.metrics import MANUAL_CALIBRATION_TOTAL, PAYLOAD_SIZE_BYTES
from depth2metric.common.settings import get_settings
from depth2metric.common.utils import get_logger
from depth2metric.inference.models import get_midas, get_yolo
from depth2metric.pipeline import depth_pcd, pack_pointcloud, precompute_samples

logger = get_logger(__name__)
settings = get_settings()

SAMPLES_DIR = Path(settings.samples_dir)
PRECOMP_DIR = Path(settings.precomputed_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once
    midas, transforms = get_midas()
    yolo = get_yolo()

    # Precompute samples (synchronous during startup is fine)
    samples_metadata = precompute_samples(midas, transforms, yolo)

    yield {
        "yolo": yolo,
        "midas": midas,
        "transforms": transforms,
        "samples_metadata": samples_metadata,
    }


app = FastAPI(
    title="Depth2Metric",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    cors.CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://depth2metric.docker-net",
        "https://depth2metric.abdelazizwf.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    trustedhost.TrustedHostMiddleware,
    allowed_hosts = [
        "abdelazizwf.dev", "*.abdelazizwf.dev",
        "localhost", "*.localhost",
        "docker-net", "*.docker-net",
    ]
)

static_files = StaticFiles(directory="static/")
static_files.is_not_modified = lambda *args, **kwargs: False # Dirty hack to disable caching
app.mount("/static", static_files, "static")

jinja = Jinja2Templates("static/")


@app.get("/health")
async def health_check():
    return None


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/telemetry/calibrate")
async def record_calibration():
    MANUAL_CALIBRATION_TOTAL.inc()
    return {"status": "ok"}


@app.get("/")
async def root(request: Request):
    samples = [p.name for p in SAMPLES_DIR.glob("*")]
    return jinja.TemplateResponse(
        "index.html",
        {
            "request": request,
            "samples": samples,
        },
    )


@app.post("/analyze/{filename}")
async def process_sample(request: Request, filename: str):
    metadata = request.state.samples_metadata.get(filename)
    if not metadata:
        raise HTTPException(404, "Sample metadata not found")

    filename_bytes = filename.split(".")[0] + ".bytes"
    path = PRECOMP_DIR / filename_bytes

    if not path.exists():
        logger.error(f"Requested sample {path!r} doesn't exist.")
        raise HTTPException(404, "Sample not found")

    # Use a thread for I/O
    loop = asyncio.get_event_loop()
    buffer = await loop.run_in_executor(None, path.read_bytes)

    # Record payload size (it's already compressed in precomputed samples)
    PAYLOAD_SIZE_BYTES.labels(type="compressed").observe(len(buffer))

    headers = dict(metadata)
    headers["Content-Encoding"] = "gzip"

    return Response(
        buffer,
        media_type="application/octet-stream",
        headers=headers,
    )


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile):
    mimes = ["image/png", "image/jpeg"]
    if file.size is None or file.content_type not in mimes:
        raise HTTPException(
            status_code=400,
            detail="Only PNG ('.png') or JPEG ('.jpg', '.jpeg') images are allowed."
        )

    if file.size > (1024 * 1024 * 8):
        raise HTTPException(
            status_code=400,
            detail="Image file is too big. Image size must be smaller than 8 MB."
        )

    loop = asyncio.get_event_loop()

    # Run heavy compute in a separate thread to keep the event loop free
    try:
        # Offload depth processing
        pcd, scale_factor, scaling_method = await loop.run_in_executor(
            None,
            partial(
                depth_pcd,
                file.file,
                request.state.midas,
                request.state.transforms,
                request.state.yolo,
            )
        )

        # Offload packing and compression
        packed_data = await loop.run_in_executor(None, pack_pointcloud, pcd)
        PAYLOAD_SIZE_BYTES.labels(type="uncompressed").observe(len(packed_data))

        result = await loop.run_in_executor(None, gzip.compress, packed_data)
        PAYLOAD_SIZE_BYTES.labels(type="compressed").observe(len(result))

    except Exception as e:
        logger.exception("Error during image analysis")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

    return Response(
        result,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": "gzip",
            "X-Scaling-Factor": str(scale_factor),
            "X-Scaling-Method": scaling_method,
        },
    )
