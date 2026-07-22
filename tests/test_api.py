import time

from tests.helpers import sharp_jpeg_bytes, blurry_jpeg_bytes, dark_jpeg_bytes, corrupted_file_bytes


def wait_until_terminal(client, image_id: str, timeout: float = 10.0) -> str:
    """Polls /status until the image reaches COMPLETED or FAILED."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{image_id}")
        status = resp.json()["status"]
        if status in ("COMPLETED", "FAILED"):
            return status
        time.sleep(0.2)
    raise TimeoutError(f"Image {image_id} did not reach a terminal state in {timeout}s")


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_returns_pending_id(client):
    resp = client.post(
        "/upload", files={"file": ("test.jpg", sharp_jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "PENDING"
    assert len(body["id"]) == 36  # UUID4 string length


def test_upload_rejects_unsupported_file_type(client):
    resp = client.post(
        "/upload", files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert resp.status_code == 415
    assert resp.json()["error"] == "unsupported_file_type"


def test_status_404_for_unknown_id(client):
    resp = client.get("/status/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"] == "image_not_found"


def test_full_pipeline_completes_and_produces_result(client):
    resp = client.post(
        "/upload", files={"file": ("clean.jpg", sharp_jpeg_bytes(), "image/jpeg")}
    )
    image_id = resp.json()["id"]

    final_status = wait_until_terminal(client, image_id)
    assert final_status == "COMPLETED"

    result = client.get(f"/result/{image_id}")
    assert result.status_code == 200
    body = result.json()
    assert body["image_id"] == image_id
    assert "confidence_score" in body
    assert body["overall_status"] in ("OK", "NEEDS_REVIEW", "REJECTED")
    assert "blur_analyzer" in body["raw_findings"]


def test_result_not_ready_before_completion_returns_409(client):
    """Immediately fetching /result after upload should return 409 or 200
    (if processing already finished, which is unlikely but possible)."""
    resp = client.post(
        "/upload", files={"file": ("quick.jpg", sharp_jpeg_bytes(), "image/jpeg")}
    )
    image_id = resp.json()["id"]
    result_resp = client.get(f"/result/{image_id}")
    # Acceptable: 409 (not ready) or 200 (already processed — fast machine).
    assert result_resp.status_code in (200, 409)
    if result_resp.status_code == 409:
        assert result_resp.json()["error"] == "analysis_not_ready"


def test_blur_analyzer_flags_blurry_image(client):
    resp = client.post(
        "/upload", files={"file": ("blurry.jpg", blurry_jpeg_bytes(), "image/jpeg")}
    )
    image_id = resp.json()["id"]
    wait_until_terminal(client, image_id)

    result = client.get(f"/result/{image_id}").json()
    assert result["is_blurry"] is True


def test_duplicate_detection_flags_repeat_upload(client):
    image_bytes = sharp_jpeg_bytes(width=500, height=500)

    first = client.post("/upload", files={"file": ("orig.jpg", image_bytes, "image/jpeg")})
    first_id = first.json()["id"]
    wait_until_terminal(client, first_id)

    second = client.post("/upload", files={"file": ("copy.jpg", image_bytes, "image/jpeg")})
    second_id = second.json()["id"]
    wait_until_terminal(client, second_id)

    result = client.get(f"/result/{second_id}").json()
    assert result["duplicate"] is True
    assert result["duplicate_of"] == first_id
    assert result["overall_status"] == "REJECTED"


def test_images_list_is_paginated(client):
    client.post("/upload", files={"file": ("a.jpg", sharp_jpeg_bytes(), "image/jpeg")})
    resp = client.get("/images?limit=1&offset=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 1
    assert len(body["items"]) <= 1
    assert body["total"] >= 1


def test_delete_image_removes_history_record(client):
    upload = client.post(
        "/upload", files={"file": ("delete-me.jpg", sharp_jpeg_bytes(width=321, height=234), "image/jpeg")}
    )
    image_id = upload.json()["id"]
    wait_until_terminal(client, image_id)

    response = client.delete(f"/images/{image_id}")
    assert response.status_code == 200
    assert response.json() == {"success": True, "deleted": 1}
    assert client.get(f"/status/{image_id}").status_code == 404


def test_stats_endpoint_returns_counts(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_images" in body
    assert "completed" in body
    assert "duplicates_detected" in body


def test_dark_image_flagged_as_low_light(client):
    resp = client.post(
        "/upload", files={"file": ("dark.jpg", dark_jpeg_bytes(), "image/jpeg")}
    )
    image_id = resp.json()["id"]
    wait_until_terminal(client, image_id)

    result = client.get(f"/result/{image_id}").json()
    assert result["is_low_light"] is True
    assert result["overall_status"] in ("NEEDS_REVIEW", "REJECTED")


def test_corrupted_image_is_handled_gracefully(client):
    """A truncated JPEG should either FAIL or COMPLETE with error findings,
    not crash the server."""
    resp = client.post(
        "/upload", files={"file": ("corrupt.jpg", corrupted_file_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 202
    image_id = resp.json()["id"]
    final_status = wait_until_terminal(client, image_id)
    # Corrupted files should reach a terminal state — either the analyzers
    # mark individual checks as errors and still produce a COMPLETED result,
    # or the whole pipeline FAILs after retries. Both are acceptable.
    assert final_status in ("COMPLETED", "FAILED")


def test_upload_rejects_oversized_file(client):
    """A file larger than MAX_UPLOAD_SIZE_MB should return 413."""
    # Generate ~11MB of data (default limit is 10MB).
    oversized = b"\x00" * (11 * 1024 * 1024)
    resp = client.post(
        "/upload", files={"file": ("huge.jpg", oversized, "image/jpeg")}
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "file_too_large"
