import gzip
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from depth2metric.inference.models import get_midas, get_yolo
from depth2metric.pipeline import depth_pcd, pack_pointcloud


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
)


static_files = StaticFiles(directory="static/")
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
    return jinja.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
def analyze(request: Request, file: UploadFile):
    mimes = ["image/png", "image/jpeg"]
    if file.content_type not in mimes:
        raise HTTPException(
            status_code=400,
            detail="Only PNG ('.png') or JPEG ('.jpg', '.jpeg') images are allowed."
        )

    # print(file.size)

    pcd = depth_pcd(
        file.file,
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
