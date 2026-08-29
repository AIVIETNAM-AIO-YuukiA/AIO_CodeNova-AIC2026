"""Media domain routes — serving frame images and per-video shot listings."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.deps import get_experiment
from api.services import media_service

router = APIRouter(tags=["media"])


@router.get("/frame")
def frame(path: str, experiment=Depends(get_experiment)):
    resolved = media_service.resolve_frame_file(experiment, path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return Response(content=resolved.read_bytes(), media_type=content_type)


@router.get("/api/video-shots")
def video_shots(video_id: str, experiment=Depends(get_experiment)):
    result, status = media_service.get_video_shots(experiment, video_id)
    if int(status) >= 400:
        raise HTTPException(status_code=int(status), detail=result.get("error", "error"))
    return result
