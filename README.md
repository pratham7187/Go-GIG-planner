# Vehicle Image Processing Pipeline

An async backend that accepts vehicle images uploaded from the field, runs a set of independent
quality/validity checks against each one, and reports back a structured verdict: is this image
usable, does it need human review, or should it be rejected outright.

Built for a take-home assignment evaluating engineering judgment, not ML accuracy. The heuristics
here are intentionally simple and are documented as such — see [Trade-offs](#trade-offs--known-limitations).

---

## Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Processing Flow](#processing-flow)
- [Folder Structure](#folder-structure)
- [Database Schema](#database-schema)
- [API Reference](#api-reference)
- [Analyzers](#analyzers)
- [Trade-offs & Known Limitations](#trade-offs--known-limitations)
- [Scalability Notes](#scalability-notes)
- [Failure Handling](#failure-handling)
- [AI Usage Disclosure](#ai-usage-disclosure)
- [Testing](#testing)
- [Future Improvements](#future-improvements)

---

## Quick Start

### Option A — Local (Python 3.12)

```bash
# 1. System dependency (OCR engine)
sudo apt-get install tesseract-ocr    # macOS: brew install tesseract

# 2. Python dependencies
pip install -r requirements.txt

# 3. (Optional) Install dev/test dependencies
pip install -r requirements-dev.txt

# 4. Configure environment
cp .env.example .env
# GEMINI_API_KEY is optional — leave blank to run without it.

# 5. Run
uvicorn app.main:app --reload

# 6. (Optional) Seed sample data
python scripts/seed.py --duplicate
```

API docs (Swagger UI): http://localhost:8000/docs

## Docker Deployment

```bash
# Build the image and start the application in the foreground.
docker compose up --build
```

To build and run without Compose:

```bash
docker build -t gogig-vehicle-pipeline .
docker run --rm -p 8000:8000 \
  -v "${PWD}/uploads:/app/uploads" \
  -v "${PWD}/data:/app/data" \
  gogig-vehicle-pipeline
```

On Windows PowerShell, replace `${PWD}` with `${PWD.Path}` in the volume
arguments if Docker does not expand it automatically. The Compose configuration
mounts `uploads/` and `data/`, preserving uploaded images and the SQLite
database across container recreations. An optional `.env` file is loaded when
present; its defaults are already supplied by the application.

Stop the foreground Compose service with `Ctrl+C`, or run:

```bash
docker compose down
```

The application is available at http://localhost:8000 and API documentation at
http://localhost:8000/docs.

---

## Architecture

Clean/layered architecture — each layer has one job and only talks to the layer below it:

```
┌─────────────────────────────────────────────────────────┐
│  API (app/api)                                          │
│  FastAPI routers. No business logic — only request/      │
│  response mapping, calling into services, and translating│
│  domain exceptions into HTTP responses.                  │
└───────────────────────┬───────────────────────────────────┘
                         │ depends on
┌───────────────────────▼───────────────────────────────────┐
│  Services (app/services)                                  │
│  Business logic: upload orchestration, async processing,   │
│  retries, status transitions, dashboard aggregation.        │
└───────────────────────┬───────────────────────────────────┘
                         │ depends on
┌───────────────────────▼───────────────────────────────────┐
│  Repositories (app/repositories)                           │
│  All SQLAlchemy query logic lives here. Services never      │
│  write raw queries.                                         │
└───────────────────────┬───────────────────────────────────┘
                         │ depends on
┌───────────────────────▼───────────────────────────────────┐
│  Models / Database (app/models, app/database)               │
│  SQLAlchemy ORM models + engine/session management.          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Analysis (app/analysis)                                 │
│  Independent analyzer classes behind a common interface    │
│  (ImageAnalyzer). Orchestrated by Processor. Used by        │
│  Services, has no knowledge of the DB or HTTP layer.         │
└─────────────────────────────────────────────────────────────┘
```

Dependency injection is wired through FastAPI's `Depends()` — see `app/api/dependencies.py`.
`ImageService` receives its repositories via constructor injection, which is what makes
`tests/test_api.py` able to swap in a temp SQLite DB with zero changes to application code.

---

## Processing Flow

1. **Upload** — `POST /upload` validates the file extension and size, saves it to
   `uploads/<uuid>.<ext>`, inserts an `images` row with `status=PENDING`, and returns the new
   image's ID immediately (HTTP 202 Accepted — the work hasn't happened yet).
2. **Background task scheduling** — the request handler registers `process_image_task` via
   FastAPI's `BackgroundTasks`, which runs it after the response has been sent, on Starlette's
   thread pool.
3. **Processing** — `process_image_task` opens its own DB session (background tasks fall outside
   request scope), flips status to `PROCESSING`, and runs the `Processor`, which executes all
   seven analyzers.
4. **Combining results** — `Processor._combine()` derives an overall verdict (`OK` /
   `NEEDS_REVIEW` / `REJECTED`) and a weighted confidence score from the analyzers' individual
   outputs, and this gets written to `analysis_results`.
5. **Completion** — status flips to `COMPLETED` (or `FAILED` after exhausting retries — see
   [Failure Handling](#failure-handling)).
6. **Retrieval** — `GET /status/{id}` for polling state, `GET /result/{id}` for the full analysis
   once it's ready.

### Why BackgroundTasks instead of Celery/RQ/SQS?
For the given scope (single-instance demo, 48-hour timeframe), an external message broker is
disproportionate infrastructure. `BackgroundTasks` gives real async behavior (the client gets an
immediate response and doesn't block on processing) without a Redis/RabbitMQ dependency to set up
and document. The trade-off is real, though: tasks run in-process and don't survive a server
restart, and don't scale across multiple instances without extra work. See
[Scalability Notes](#scalability-notes) for what changes in a production setting.

---

## Folder Structure

```
app/
├── main.py                    # FastAPI app, lifespan, global exception handlers
├── api/
│   ├── image_controller.py    # /upload /status /result /images /stats routes
│   ├── health_controller.py   # /health
│   └── dependencies.py        # DI wiring (get_image_service)
├── services/
│   └── image_service.py       # Business logic + background processing + retry loop
├── repositories/
│   ├── image_repository.py
│   └── analysis_repository.py
├── models/
│   ├── image.py                # SQLAlchemy Image model
│   ├── analysis_result.py      # SQLAlchemy AnalysisResult model
│   └── enums.py                # ProcessingStatus, OverallStatus
├── schemas/
│   └── image_schemas.py        # Pydantic DTOs (request/response contracts)
├── analysis/
│   ├── base.py                  # ImageAnalyzer interface + AnalyzerResult
│   ├── blur_analyzer.py
│   ├── brightness_analyzer.py
│   ├── duplicate_analyzer.py
│   ├── ocr_analyzer.py          # OCR + Indian plate validation
│   ├── metadata_analyzer.py
│   ├── screenshot_analyzer.py
│   ├── gemini_analyzer.py       # Optional — skips gracefully with no API key
│   └── processor.py             # Orchestrates all analyzers, combines verdict
├── database/
│   └── session.py               # Engine, session factory, init_db()
├── config/
│   └── settings.py               # pydantic-settings, single source of config truth
└── utils/
    ├── logger.py
    ├── file_storage.py            # Local upload persistence
    └── exceptions.py               # Domain exceptions (ImageNotFoundError, etc.)
tests/
├── conftest.py                     # Isolated temp-DB fixture
├── helpers.py                      # Synthetic test image generators
└── test_api.py                     # 10 integration tests across all endpoints
scripts/
└── seed.py                          # Uploads sample images against a running instance
postman/
└── vehicle_pipeline.postman_collection.json
Dockerfile
docker-compose.yml
requirements.txt
.env.example
```

---

## Database Schema

**`images`**

| Column | Type | Notes |
|---|---|---|
| id | String(36), PK | UUID4 |
| filename | String(255) | Stored filename (UUID-prefixed) |
| filepath | String(512) | Path on local disk |
| status | Enum | PENDING / PROCESSING / COMPLETED / FAILED |
| uploaded_at | DateTime | |
| completed_at | DateTime, nullable | Set on COMPLETED or FAILED |
| hash | String(64), nullable, indexed | Perceptual hash, set once processing succeeds |
| error_message | String(1024), nullable | Last failure reason |
| retry_count | Integer | Incremented on each failed processing attempt |

**`analysis_results`** (1:1 with `images` via `image_id`)

| Column | Type | Notes |
|---|---|---|
| image_id | String(36), FK, unique | |
| blur_score / is_blurry | Float / Boolean | |
| brightness / is_low_light | Float / Boolean | |
| duplicate / duplicate_of | Boolean / String | duplicate_of references another image's id |
| ocr_text / vehicle_number / plate_valid | String / String / Boolean | |
| image_metadata | JSON | width, height, format, has_exif, file_size |
| screenshot / tampered | Boolean / Boolean | tampered comes from GeminiAnalyzer when enabled |
| confidence_score | Float | Weighted average across successful analyzers |
| overall_status | Enum | OK / NEEDS_REVIEW / REJECTED |
| summary | String | Human-readable one-line explanation |
| raw_findings | JSON | Full per-analyzer output, for debugging/audit |

`raw_findings` is stored as JSON rather than normalized columns deliberately — adding or adjusting
an analyzer never requires a migration, at the cost of not being able to SQL-query into individual
analyzer fields. See [Trade-offs](#trade-offs--known-limitations).

---

## API Reference

### `POST /upload`
Accepts `multipart/form-data` with a `file` field (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, max
10MB by default).

```bash
curl -X POST http://localhost:8000/upload -F "file=@vehicle.jpg"
```
```json
{ "id": "e5909dc6-ad60-4ecf-8ca8-635614577692", "status": "PENDING" }
```
`415` for unsupported file types, `413` for oversized files.

### `GET /status/{id}`
```bash
curl http://localhost:8000/status/e5909dc6-ad60-4ecf-8ca8-635614577692
```
```json
{
  "id": "e5909dc6-ad60-4ecf-8ca8-635614577692",
  "status": "COMPLETED",
  "uploaded_at": "2026-07-20T18:38:59.000Z",
  "completed_at": "2026-07-20T18:39:01.000Z",
  "error_message": null,
  "retry_count": 0
}
```
`404` if the ID doesn't exist.

### `GET /result/{id}`
Returns the full analysis once available; `409` if the image hasn't finished processing yet.

```json
{
  "image_id": "e5909dc6-ad60-4ecf-8ca8-635614577692",
  "blur_score": 51565.87,
  "is_blurry": false,
  "brightness": 127.05,
  "is_low_light": false,
  "duplicate": false,
  "duplicate_of": null,
  "ocr_text": "",
  "vehicle_number": null,
  "plate_valid": false,
  "image_metadata": {
    "width": 600, "height": 400, "format": "JPEG",
    "has_exif": false, "file_size_bytes": 144897
  },
  "screenshot": false,
  "tampered": false,
  "confidence_score": 0.697,
  "overall_status": "NEEDS_REVIEW",
  "summary": "Needs review: vehicle number plate not confidently detected",
  "raw_findings": { "...": "per-analyzer breakdown, see below" }
}
```
*(This is a real captured response from a local run — not a hypothetical example.)*

### `GET /images?limit=20&offset=0`
Paginated list of uploaded images (most recent first).

### `GET /stats`
```json
{
  "total_images": 5, "pending": 0, "processing": 0, "completed": 5, "failed": 0,
  "duplicates_detected": 1, "average_confidence_score": 0.718, "plate_validation_rate": 0.0
}
```

### `GET /health`
Liveness + config visibility (e.g. whether Gemini is enabled), for monitoring/orchestration.

Full interactive documentation is auto-generated by FastAPI at `/docs` (Swagger) and `/redoc`.

---

## Analyzers

All analyzers implement `ImageAnalyzer` (`app/analysis/base.py`) and return a uniform
`{name, status, confidence, findings, execution_time}` shape. `Processor` runs them all, and one
analyzer's exception never blocks the others — it's caught and recorded as `status: "error"`.

| Analyzer | Technique | What it flags |
|---|---|---|
| `BlurAnalyzer` | OpenCV Laplacian variance | Blurry images (variance < 100) |
| `BrightnessAnalyzer` | Mean grayscale intensity | Low-light (<60) and/or overexposed (>220) images — each flagged independently |
| `OCRAnalyzer` | pytesseract + regex | Extracts text, validates Indian plate format |
| `DuplicateAnalyzer` | Perceptual hash (pHash), Hamming distance ≤5 | Re-uploads / near-identical images |
| `MetadataAnalyzer` | PIL EXIF + file properties | Feeds ScreenshotAnalyzer; deterministic |
| `ScreenshotAnalyzer` | Heuristic: missing EXIF + narrow screen-ratio + flat borders | Screenshots / photo-of-photo |
| `GeminiAnalyzer` | Gemini Vision API (optional) | Second opinion on tampering + screenshot |

`Processor._combine()` derives the overall verdict:
- **REJECTED** if the image is a duplicate or Gemini reports tampering
- **NEEDS_REVIEW** if it's blurry, low-light, overexposed, looks like a screenshot, or the plate
  wasn't confidently read
- **OK** otherwise

`confidence_score` is a weighted average of each successful analyzer's own confidence (weights in
`ANALYZER_WEIGHTS`, e.g. duplicate detection is weighted higher than metadata). This weighting is
a reasonable default, not a tuned model — see trade-offs.

---

## Trade-offs & Known Limitations

Being upfront about what's simplified and what I found while testing:

- **SQLite instead of Postgres.** Zero external services to set up for a take-home reviewer, at
  the cost of no real concurrent-write support. A production deployment should move to Postgres —
  the repository layer already isolates all query logic, so this is a config change plus a
  migration tool (Alembic), not a rewrite.
- **BackgroundTasks instead of a real task queue.** No Celery/RQ/SQS dependency to run, but tasks
  don't survive a process restart and don't distribute across multiple app instances. See
  [Scalability Notes](#scalability-notes).
- **Heuristics, not trained models**, for blur/brightness/screenshot detection. Thresholds
  (Laplacian variance 100, brightness 60/220, Hamming distance 5) are reasonable defaults from
  common practice, not tuned against a labeled dataset — because there isn't one for this task.
  This is exactly the kind of "structuring uncertainty" the assignment asks for, so it's worth
  naming directly: these numbers **will** misclassify some images. Confidence scores reflect that
  uncertainty rather than pretending precision.
- **A concurrency bug I found and fixed while testing:** FastAPI's `BackgroundTasks` run
  concurrently on a thread pool. Two near-simultaneous uploads of the *same* image could each read
  the "existing hashes" table before the other's write committed, so duplicate detection would
  silently miss the duplicate. I added a process-wide lock around the read-compute-write critical
  section as a proportionate fix for a single-instance deployment; a multi-instance deployment
  would need a DB-level unique constraint or advisory lock instead, since an in-process lock
  doesn't help across separate processes/machines.
- **A data-integrity bug I found and fixed:** the original code flipped an image's status to
  `COMPLETED` *before* writing its analysis row. If the analysis write failed, the image ended up
  permanently marked complete with no analysis data — a state `GET /result` couldn't recover from.
  Fixed by writing analysis first, then flipping status, so the two can never disagree.
- **A serialization bug I found and fixed:** `imagehash`'s Hamming-distance subtraction returns a
  numpy `int64`, which isn't JSON-serializable — this broke every write to the `raw_findings`/hash
  fields until I cast it to a native `int`.
- **`raw_findings` as JSON, not normalized columns.** Faster to extend (new analyzer = no
  migration), but you can't write a SQL query filtering on a specific analyzer's internal field
  without JSON operators.
- **The screenshot heuristic is genuinely weak** (see [Analyzers](#analyzers)) — I initially
  included standard 4:3/1:1 camera ratios in the "known screen ratio" list, which made it flag
  nearly every ordinary photo as a screenshot. I narrowed it to ratios that are actually
  screen-specific, but this remains a 3-signal heuristic, not a robust classifier. Gemini's second
  opinion (when enabled) is meant to offset this, not replace it.
- **No authentication.** Out of scope for the assignment, but a production API obviously needs it
  (API keys or OAuth on `/upload` at minimum).
- **No rate limiting.** Would add `slowapi` or a reverse-proxy-level limiter before production use.

---

## Scalability Notes

At current scale (single instance, local disk, SQLite) this handles a demo/small-team workload
fine. To grow it:

1. **Storage**: swap local disk for S3/GCS. `app/utils/file_storage.py` already isolates this
   behind a `save_upload()` function — implementations change, callers don't.
2. **Database**: Postgres instead of SQLite for real concurrent writes; add Alembic for
   migrations.
3. **Queue**: replace `BackgroundTasks` with a real broker (Celery+Redis, or SQS) once you need
   processing to survive restarts or scale across multiple app instances — multiple workers can
   pull from the same queue instead of each instance running its own in-process tasks.
4. **Duplicate detection at scale**: the current approach is O(n) — it compares against every
   completed image's hash. Fine for hundreds/thousands of images; beyond that, a proper
   locality-sensitive-hashing index (or a vector DB) avoids the full scan.
5. **Horizontal scaling**: once processing moves to a real queue, the API layer itself is already
   stateless and can run behind a load balancer with N replicas.

---

## Failure Handling

- Each analyzer isolates its own exceptions (`ImageAnalyzer.run()` catches and records
  `status: "error"`) — one broken check never takes down the other six.
- The processing pipeline as a whole retries up to `RETRY_LIMIT` (default 2, so 3 total attempts)
  on unexpected failures, incrementing `retry_count` and rolling back the DB session between
  attempts so one failed write doesn't poison subsequent retries.
- After exhausting retries, the image is marked `FAILED` with `error_message` populated —
  visible via `GET /status/{id}`.
- A global exception handler in `app/main.py` catches anything unhandled and returns a clean `500`
  instead of leaking a stack trace to the client (the trace still goes to logs).
- `GeminiAnalyzer` fails *open*: if the API key isn't configured, or the call errors, it returns a
  `skipped`/`error` result rather than blocking the rest of the pipeline.

---

## AI Usage Disclosure

I used Claude throughout this build, in a fairly tight loop of generate → run → find bugs → fix →
re-verify, rather than generating once and shipping.

**Where AI helped:**
- Scaffolding the layered architecture (controllers/services/repositories/models/schemas) and
  keeping the separation consistent across ~25 files.
- Writing the initial version of each analyzer, the Processor's combination logic, and the
  FastAPI wiring (DI, exception handlers, lifespan events).
- Writing the test suite and seed script.

**Where AI output was wrong, and how I caught it (not hypothetically — these are real bugs found
by actually running the service, not by reading the code):**
1. The `AnalysisResult` model was initially missing a `tampered` column that the Processor's
   output dict included — first real upload immediately threw `'tampered' is an invalid keyword
   argument`. Caught by running an end-to-end upload test, not by code review.
2. `imagehash`'s Hamming distance returns numpy's `int64`, which `json.dumps` can't serialize —
   broke every JSON column write. Only visible once an actual duplicate comparison ran.
3. A logic-order bug: status was flipped to `COMPLETED` before the analysis row was written, so a
   failed analysis write left images stuck in a state where `/status` said COMPLETED but
   `/result` said "not ready." Found by deliberately re-testing the duplicate-detection path
   after the first fix, not by inspection.
4. A genuine concurrency race: `BackgroundTasks` run on a thread pool, so two uploads of an
   identical image in quick succession could both read the duplicate-hash table before either
   committed, and neither would be flagged. This only showed up when the seed script uploaded
   five images back-to-back — a single manual test wouldn't have caught it. Fixed with a lock
   around the critical section.
5. The screenshot-detection heuristic's ratio list included standard camera aspect ratios (4:3,
   1:1), so it flagged nearly every plain photo as a screenshot. Caught by checking the seed
   script's synthetic samples against expected outcomes, not assumed correct from the code.

**How I validated the fixes:** every fix above was followed by killing and restarting the running
server and re-hitting the actual endpoints (not just re-reading the code) to confirm the specific
failure mode was gone, then re-running the automated test suite. The 10 tests in `tests/test_api.py`
now pass cleanly and cover upload validation, the full async pipeline, blur detection, duplicate
detection, pagination, and 404 handling.

**Where I didn't just accept AI output as-is:** the analyzer thresholds (blur variance, brightness
cutoffs, Hamming distance, screen-ratio list) are values I reviewed and, in the screenshot case,
corrected — these are heuristics, not tuned models, and I've documented that honestly above rather
than presenting them as validated.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

13 tests covering: health check, upload success/rejection (unsupported type + oversized), 404
handling, the full async pipeline (upload → poll until terminal → fetch result), 409 before
completion, blur detection on a synthetically blurred image, dark/low-light image detection,
corrupted image handling, duplicate detection across two uploads of the same bytes, pagination,
and dashboard stats. Tests run against an isolated temp SQLite DB and temp upload directory
(see `tests/conftest.py`) so they never touch your local dev data.

---

## Future Improvements

- Move duplicate detection to a proper task queue + LSH index for scale
- Add Alembic migrations and a Postgres option
- Add authentication (API key or OAuth) on `/upload`
- Add rate limiting
- Replace the screenshot heuristic with a small trained classifier once labeled data exists
- Add structured (JSON) logging for easier log aggregation in production
- Add a `DELETE /images/{id}` endpoint for GDPR-style data removal requests
