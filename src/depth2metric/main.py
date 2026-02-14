import gzip
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware import cors, trustedhost
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from depth2metric.inference.models import get_midas, get_yolo
from depth2metric.pipeline import depth_pcd, pack_pointcloud

SAMPLES_DIR = Path("static/samples")


@asynccontextmanager
async def lifespan(app: FastAPI):
    midas, transforms = get_midas()
    yolo = get_yolo()
    yield {
        "yolo": yolo,
        "midas": midas,
        "transforms": transforms,
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
def process_sample(request: Request, filename: str):
    path = SAMPLES_DIR / filename

    if not path.exists():
        raise HTTPException(404, "Sample not found")

    with open(path, "rb") as file:
        pcd = depth_pcd(
            file,
            request.state.midas,
            request.state.transforms,
            request.state.yolo,
        )

        result = gzip.compress(pack_pointcloud(pcd))

        return Response(
            result,
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip"},
        )


@app.post("/analyze")
def analyze(request: Request, file: UploadFile):
    mimes = ["image/png", "image/jpeg"]
    if file.size is None or file.content_type not in mimes:
        raise HTTPException(
            status_code=400,
            detail="Only PNG ('.png') or JPEG ('.jpg', '.jpeg') images are allowed."
        )

    if file.size > (1024 * 1024 * 8):
        raise HTTPException(
            status_code=400,
            detail="Image file is too big. Image size must be smaller that 8 MB."
        )

    pcd = depth_pcd(
        file.file,
        request.state.midas,
        request.state.transforms,
        request.state.yolo,
    )

    file.file.close()

    result = gzip.compress(pack_pointcloud(pcd))

    return Response(
        result,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )
