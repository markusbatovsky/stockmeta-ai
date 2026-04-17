"""
server.py   -   StockMeta AI  v3.0
----------------------------------------------------------------------
Database  : SQLite (built-in, zero setup)
AI        : OpenAI GPT-4o (preferred) or Google Gemini 2.0 Flash
Background: threads (built-in)
Requires  : fastapi  uvicorn  openai  google-generativeai  pillow  python-multipart  aiosqlite
"""

import asyncio
import base64
import csv
import io
import json
import logging
import mimetypes
import os
import struct
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import aiosqlite
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "stockmeta.db"
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND   = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AI_MODEL       = os.getenv("AI_MODEL", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("stockmeta")

# -- AI provider setup ----------------------------------------------------------
if OPENAI_API_KEY:
    import openai as _openai
    _oa_client = _openai.OpenAI(api_key=OPENAI_API_KEY)
    _oa_model  = AI_MODEL or "gpt-4o"
    _provider  = "OpenAI"
elif GOOGLE_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    _gemini_model = AI_MODEL or "gemini-2.0-flash"
    _provider = "Google Gemini"
else:
    _provider = "none"
    log.warning("No AI provider configured  -  set OPENAI_API_KEY or GOOGLE_API_KEY")

# -- Getty ESP keyword prompt ---------------------------------------------------
GETTY_PROMPT = """You are a Getty Images editorial and creative content specialist with deep expertise in stock photography keywording using the ESP (Editorial Standards & Practices) system.

Analyze this image carefully and generate professional stock metadata optimized for maximum discoverability on Getty Images, Shutterstock, Adobe Stock, Alamy, and Pond5.

RULES:
Title:
- 5-10 words, Title Case noun phrase
- Describe the SPECIFIC subject, NOT generic scene
- No keyword stuffing, no comma lists

Description:
- 1-2 sentences describing scene, mood, composition, lighting, subject
- Mention visual style, color palette, emotional tone
- Commercially relevant language

Keywords (Getty ESP standard):
- Generate exactly 30-50 keywords
- Order by commercial relevance (most searchable first)
- Include: main subject, action/pose, location/setting, mood/emotion, color, style, demographics (if people), concepts, industry/use case
- SPECIFIC over GENERIC: "golden retriever" not "dog", "businessman in suit" not "person"
- Include both singular and plural where relevant
- Include conceptual keywords buyers search: "teamwork", "success", "freedom", "sustainability"
- NO banned words: image, photo, stock, picture, concept (standalone), background (standalone)
- Mix concrete (what you see) + abstract (what it means)

Category:
Choose ONE: Abstract | Animals/Wildlife | Arts | Backgrounds/Textures | Beauty/Fashion | Buildings/Landmarks | Business/Finance | Education | Food/Drink | Healthcare/Medical | Holidays | Industrial | Nature | Objects | Parks/Outdoor | People | Science | Sports/Recreation | Technology | Transportation | Travel

Respond ONLY with this exact JSON (no markdown, no explanation):
{
  "title": "...",
  "description": "...",
  "keywords": ["kw1", "kw2", "..."],
  "category": "..."
}"""


# -- Database -------------------------------------------------------------------
async def get_db():
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    return db

async def init_db():
    db = await get_db()
    await db.executescript("""
    CREATE TABLE IF NOT EXISTS assets (
        id            TEXT PRIMARY KEY,
        filename      TEXT NOT NULL,
        original_name TEXT NOT NULL,
        asset_type    TEXT NOT NULL,
        mime_type     TEXT NOT NULL,
        file_size     INTEGER,
        width         INTEGER,
        height        INTEGER,
        duration      REAL,
        storage_path  TEXT,
        status        TEXT DEFAULT 'ingested',
        error_msg     TEXT,
        hints         TEXT,
        exif_data     TEXT,
        color_palette TEXT,
        collection_id TEXT,
        created_at    TEXT DEFAULT (datetime('now')),
        updated_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS metadata_outputs (
        id              TEXT PRIMARY KEY,
        asset_id        TEXT NOT NULL,
        platform        TEXT NOT NULL DEFAULT 'getty',
        version         INTEGER DEFAULT 1,
        title           TEXT,
        description     TEXT,
        keywords        TEXT,
        category        TEXT,
        is_active       INTEGER DEFAULT 1,
        manually_edited INTEGER DEFAULT 0,
        created_at      TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS collections (
        id         TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        brief      TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """)
    # Migrate old DBs  -  add missing columns silently
    for col_sql in [
        "ALTER TABLE assets ADD COLUMN exif_data TEXT",
        "ALTER TABLE assets ADD COLUMN color_palette TEXT",
        "ALTER TABLE assets ADD COLUMN collection_id TEXT",
    ]:
        try:
            await db.execute(col_sql)
            await db.commit()
        except Exception:
            pass
    await db.commit()
    await db.close()


# -- EXIF extraction -----------------------------------------------------------
def extract_exif(path: Path) -> dict:
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS
        img = PILImage.open(str(path))
        raw = img._getexif() or {}
        tag_map = {TAGS.get(k, k): v for k, v in raw.items()}

        def _rational(v):
            try:
                if hasattr(v, 'numerator'):
                    return round(v.numerator / v.denominator, 4)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    return round(v[0] / v[1], 4) if v[1] else None
            except Exception:
                pass
            return v

        exif = {}
        if "Make" in tag_map and "Model" in tag_map:
            make = str(tag_map["Make"]).strip().rstrip('\x00')
            model = str(tag_map["Model"]).strip().rstrip('\x00')
            exif["camera"] = f"{make} {model}".strip() if make.lower() not in model.lower() else model
        elif "Model" in tag_map:
            exif["camera"] = str(tag_map["Model"]).strip().rstrip('\x00')

        if "LensModel" in tag_map:
            exif["lens"] = str(tag_map["LensModel"]).strip().rstrip('\x00')

        if "FocalLength" in tag_map:
            fl = _rational(tag_map["FocalLength"])
            if fl: exif["focal_length"] = f"{int(fl)}mm"

        if "FNumber" in tag_map:
            fn = _rational(tag_map["FNumber"])
            if fn: exif["aperture"] = f"f/{fn}"

        if "ISOSpeedRatings" in tag_map:
            exif["iso"] = str(tag_map["ISOSpeedRatings"])

        if "ExposureTime" in tag_map:
            et = _rational(tag_map["ExposureTime"])
            if et:
                if et < 1:
                    exif["shutter_speed"] = f"1/{int(round(1/et))}s"
                else:
                    exif["shutter_speed"] = f"{et}s"

        if "DateTimeOriginal" in tag_map:
            exif["capture_date"] = str(tag_map["DateTimeOriginal"])

        return exif
    except Exception as e:
        log.debug("EXIF extraction failed: %s", e)
        return {}


# -- Color palette extraction --------------------------------------------------
def _rgb_to_hsv(r: int, g: int, b: int):
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r_, g_, b_)
    cmin = min(r_, g_, b_)
    delta = cmax - cmin
    s = 0 if cmax == 0 else delta / cmax
    return cmax, s  # return value (brightness) and saturation


def extract_palette(path: Path, n_colors: int = 6) -> list:
    """Extract dominant colors, filtering out near-black/near-white background pixels
    and sorting by saturation so studio backgrounds don't dominate."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(str(path)).convert("RGB")
        img.thumbnail((200, 200))
        colors = img.getcolors(maxcolors=200 * 200)
        if not colors:
            return []

        # Filter: skip near-black (<30 avg) and near-white (>220 avg) pixels
        # which are typically studio backgrounds
        filtered = []
        for count, rgb in colors:
            r, g, b = rgb
            avg = (r + g + b) / 3
            if avg < 28 or avg > 222:
                continue
            v, s = _rgb_to_hsv(r, g, b)
            filtered.append((count, rgb, s, v))

        # If everything was filtered (e.g. pure B&W photo) fall back to all colors
        if not filtered:
            filtered = [(count, rgb, *_rgb_to_hsv(*rgb)) for count, rgb in colors
                        if (sum(rgb) / 3) >= 28]

        # Sort: higher saturation first, then by count
        filtered.sort(key=lambda x: (-x[2], -x[0]))

        palette = []
        for count, rgb, s, v in filtered:
            r, g, b = rgb
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            # skip if too similar to existing
            duplicate = False
            for existing in palette:
                er = int(existing[1:3], 16)
                eg = int(existing[3:5], 16)
                eb = int(existing[5:7], 16)
                if abs(r - er) + abs(g - eg) + abs(b - eb) < 55:
                    duplicate = True
                    break
            if not duplicate:
                palette.append(hex_color)
            if len(palette) >= n_colors:
                break
        return palette
    except Exception as e:
        log.debug("Color palette extraction failed: %s", e)
        return []


# -- AI analysis ---------------------------------------------------------------
def analyze_image_sync(asset_id: str, image_path: str, hints: dict | None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_analyze_and_save(asset_id, image_path, hints))
    finally:
        loop.close()

async def _analyze_and_save(asset_id: str, image_path: str, hints: dict | None):
    db = await get_db()
    try:
        await db.execute("UPDATE assets SET status='processing', updated_at=datetime('now') WHERE id=?", (asset_id,))
        await db.commit()

        from PIL import Image as PILImage
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        img = PILImage.open(str(path)).convert("RGB")

        # Extract EXIF and color palette
        exif_data = extract_exif(path)
        color_palette = extract_palette(path)

        # Build prompt: brief goes FIRST as mandatory override, other context appended after
        brief_block = ""
        context_parts = []

        if hints:
            b = hints.get("brief", "").strip()
            cb = hints.get("collection_brief", "").strip()
            combined_brief = "\n".join(filter(None, [b, cb]))
            if combined_brief:
                brief_block = (
                    "=== MANDATORY PHOTOGRAPHER BRIEF ===\n"
                    "The photographer provided the following instructions. You MUST incorporate "
                    "this theme/context heavily into the title, description, and especially keywords. "
                    "This brief defines the commercial angle - do not ignore it.\n\n"
                    f"BRIEF: {combined_brief}\n"
                    "=== END BRIEF ===\n\n"
                )
            if hints.get("location"):
                context_parts.append(f"Location: {hints['location']}")
            if hints.get("shoot_type"):
                context_parts.append(f"Shoot type: {hints['shoot_type']}")
            if hints.get("people_count") is not None:
                context_parts.append(f"People count: {hints['people_count']}")
            if hints.get("intent"):
                context_parts.append(f"Usage intent: {hints['intent']}")

        context_text = ("\n\n---\nAdditional context:\n" + "\n".join(context_parts)) if context_parts else ""
        prompt = brief_block + GETTY_PROMPT + context_text

        # Call AI
        if OPENAI_API_KEY:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            rsp = _oa_client.chat.completions.create(
                model=_oa_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
                ]}],
                max_tokens=1500,
            )
            raw = rsp.choices[0].message.content.strip()
        elif GOOGLE_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(AI_MODEL or "gemini-2.0-flash")
            response = model.generate_content([prompt, img])
            raw = response.text.strip()
        else:
            raise ValueError("No AI provider configured")

        # Parse JSON response
        if "```" in raw:
            parts_split = raw.split("```")
            raw = parts_split[1] if len(parts_split) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())

        await db.execute("UPDATE metadata_outputs SET is_active=0 WHERE asset_id=?", (asset_id,))
        await db.execute(
            "INSERT INTO metadata_outputs (id, asset_id, platform, title, description, keywords, category) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), asset_id, "getty",
             data.get("title", ""), data.get("description", ""),
             json.dumps(data.get("keywords", [])), data.get("category", ""))
        )
        await db.execute(
            "UPDATE assets SET status='complete', updated_at=datetime('now'), exif_data=?, color_palette=? WHERE id=?",
            (json.dumps(exif_data), json.dumps(color_palette), asset_id)
        )
        await db.commit()
        log.info("Asset %s complete", asset_id)
    except Exception as e:
        log.error("Asset %s failed: %s", asset_id, e)
        await db.execute(
            "UPDATE assets SET status='error', error_msg=?, updated_at=datetime('now') WHERE id=?",
            (str(e), asset_id)
        )
        await db.commit()
    finally:
        await db.close()


# -- FastAPI app ----------------------------------------------------------------
app = FastAPI(title="StockMeta AI", version="3.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi"}


# -- Asset endpoints ------------------------------------------------------------
@app.post("/api/v1/assets/upload", status_code=202)
async def upload_asset(
    file: UploadFile = File(...),
    hints: Optional[str] = Form(None),
    collection_id: Optional[str] = Form(None),
):
    ext = Path(file.filename or "file").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {ext}")
    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 500MB)")

    asset_id  = str(uuid.uuid4())
    safe_name = f"{asset_id}{ext}"
    store_path = str(UPLOAD_DIR / safe_name)
    with open(store_path, "wb") as f:
        f.write(content)

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "image/jpeg"
    asset_type = "image" if mime.startswith("image/") else "video"

    width = height = None
    if asset_type == "image":
        try:
            from PIL import Image as PILImage
            with PILImage.open(store_path) as img:
                width, height = img.size
        except Exception:
            pass

    hints_dict = {}
    if hints:
        try:
            hints_dict = json.loads(hints)
        except Exception:
            hints_dict = {"brief": hints}

    # Attach collection brief if collection_id provided
    if collection_id:
        db_temp = await get_db()
        try:
            coll = await (await db_temp.execute(
                "SELECT brief FROM collections WHERE id=?", (collection_id,)
            )).fetchone()
            if coll and coll["brief"]:
                hints_dict["collection_brief"] = coll["brief"]
        finally:
            await db_temp.close()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO assets (id,filename,original_name,asset_type,mime_type,file_size,width,height,"
            "storage_path,status,hints,collection_id) VALUES (?,?,?,?,?,?,?,?,?,'ingested',?,?)",
            (asset_id, safe_name, file.filename, asset_type, mime, len(content),
             width, height, store_path,
             json.dumps(hints_dict) if hints_dict else None,
             collection_id)
        )
        await db.commit()
    finally:
        await db.close()

    t = threading.Thread(
        target=analyze_image_sync,
        args=(asset_id, store_path, hints_dict),
        daemon=True
    )
    t.start()

    return {
asset_id": asset_id, "filename": file.filename, "asset_type": asset_type,
        "status": "ingested", "message": "File accepted. AI generating metadata...",
        "estimated_processing_seconds": 20
    }


@app.get("/api/v1/assets/{asset_id}")
async def get_asset(asset_id: str):
    db = await get_db()
    try:
        row = await (await db.execute("SELECT * FROM assets WHERE id=?", (asset_id,))).fetchone()
        if not row: raise HTTPException(404, "Asset not found")
        return _asset_dict(row)
    finally:
        await db.close()


@app.get("/api/v1/assets/")
async def list_assets(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    status: Optional[str] = Query(None),
    asset_type: Optional[str] = Query(None),
    collection_id: Optional[str] = Query(None),
):
    db = await get_db()
    try:
        where, params = "WHERE 1=1", []
        if status:        where += " AND status=?";        params.append(status)
        if asset_type:    where += " AND asset_type=?";    params.append(asset_type)
        if collection_id: where += " AND collection_id=?"; params.append(collection_id)
        total = (await (await db.execute(f"SELECT COUNT(*) FROM assets {where}", params)).fetchone())[0]
        rows  = await (await db.execute(
            f"SELECT * FROM assets {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [page_size, (page-1)*page_size]
        )).fetchall()
        return {"items": [_asset_dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}
    finally:
        await db.close()


@app.delete("/api/v1/assets/{asset_id}", status_code=204)
async def delete_asset(asset_id: str):
    db = await get_db()
    try:
        row = await (await db.execute("SELECT storage_path FROM assets WHERE id=?", (asset_id,))).fetchone()
        if not row: raise HTTPException(404, "Asset not found")
        try: Path(row["storage_path"]).unlink(missing_ok=True)
        except: pass
        await db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
        await db.commit()
    finally:
        await db.close()


def _asset_dict(row):
    return {
        "asset_id": row["id"], "filename": row["original_name"],
        "asset_type": row["asset_type"], "status": row["status"],
        "width": row["width"], "height": row["height"],
        "duration_secs": row["duration"], "file_size_bytes": row["file_size"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "error_message": row["error_msg"],
        "exif_data": json.loads(row["exif_data"]) if row["exif_data"] else {},
        "color_palette": json.loads(row["color_palette"]) if row["color_palette"] else [],
        "collection_id": row["collection_id"],
    }


# -- Metadata endpoints ---------------------------------------------------------
@app.get("/api/v1/metadata/{asset_id}")
async def get_metadata(asset_id: str):
    db = await get_db()
    try:
        asset = await (await db.execute("SELECT * FROM assets WHERE id=?", (asset_id,))).fetchone()
        if not asset: raise HTTPException(404, "Asset not found")
        if asset["status"] not in ("complete",):
            raise HTTPException(409, f"Status is '{asset['status']}'")
        row = await (await db.execute(
            "SELECT * FROM metadata_outputs WHERE asset_id=? AND is_active=1 ORDER BY created_at DESC LIMIT 1",
            (asset_id,)
        )).fetchone()
        if not row: raise HTTPException(404, "No metadata found")
        kws = json.loads(row["keywords"] or "[]")
        return {
            "asset_id": asset_id,
            "asset_type": asset["asset_type"],
            "exif_data": json.loads(asset["exif_data"]) if asset["exif_data"] else {},
            "color_palette": json.loads(asset["color_palette"]) if asset["color_palette"] else [],
            "getty": {
                "title": row["title"],
                "description": row["description"],
                "keywords_ranked": kws,
                "top10_keywords": kws[:10],
                "category_primary": row["category"],
                "version": row["version"]
            }
        }
    finally:
        await db.close()


@app.put("/api/v1/metadata/{asset_id}")
async def edit_metadata(asset_id: str, body: dict):
    db = await get_db()
    try:
        row = await (await db.execute(
            "SELECT id FROM metadata_outputs WHERE asset_id=? AND is_active=1 ORDER BY created_at DESC LIMIT 1",
            (asset_id,)
        )).fetchone()
        if not row: raise HTTPException(404, "Metadata not found")
        updates = {}
        if "title"           in body: updates["title"]       = body["title"]
        if "description"     in body: updates["description"] = body["description"]
        if "keywords_ranked" in body: updates["keywords"]    = json.dumps(body["keywords_ranked"])
        if "category_primary"in body: updates["category"]    = body["category_primary"]
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            await db.execute(
                f"UPDATE metadata_outputs SET {set_clause}, manually_edited=1 WHERE id=?",
                list(updates.values()) + [row["id"]]
            )
            await db.commit()
        return {"status": "updated", "asset_id": asset_id}
    finally:
        await db.close()


@app.post("/api/v1/metadata/{asset_id}/regenerate", status_code=202)
async def regenerate_metadata(asset_id: str, hints: Optional[dict] = None):
    db = await get_db()
    try:
        asset = await (await db.execute("SELECT * FROM assets WHERE id=?", (asset_id,))).fetchone()
        if not asset: raise HTTPException(404, "Asset not found")
        await db.execute("UPDATE metadata_outputs SET is_active=0 WHERE asset_id=?", (asset_id,))
        await db.execute("UPDATE assets SET status='ingested', updated_at=datetime('now') WHERE id=?", (asset_id,))
        await db.commit()
        stored_hints = json.loads(asset["hints"] or "{}")
        if hints:
            stored_hints.update(hints)
    finally:
        await db.close()
    t = threading.Thread(
        target=analyze_image_sync,
        args=(asset_id, str(UPLOAD_DIR / asset["filename"]), stored_hints),
        daemon=True
    )
    t.start()
    return {"status": "queued", "message": "Regeneration started"}


# -- CSV Export (MicrostockPlus format) ----------------------------------------
@app.post("/api/v1/export/csv")
async def export_csv(asset_ids: Optional[List[str]] = Query(None)):
    """Export in MicrostockPlus format: Filename,Title,Description,Keywords,Category"""
    db = await get_db()
    try:
        where, params = "WHERE a.status='complete' AND m.is_active=1", []
        if asset_ids:
            ph = ",".join("?" * len(asset_ids))
            where += f" AND a.id IN ({ph})"
            params.extend(asset_ids)
        rows = await (await db.execute(
            f"SELECT a.original_name, m.title, m.description, m.keywords, m.category "
            f"FROM assets a JOIN metadata_outputs m ON m.asset_id=a.id "
            f"{where} ORDER BY a.created_at DESC",
            params
        )).fetchall()
        if not rows:
            raise HTTPException(404, "No completed assets found")

        output = io.StringIO()
        w = csv.writer(output, delimiter=",", quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["Filename", "Title", "Description", "Keywords", "Category"])
        for r in rows:
            kws = json.loads(r["keywords"] or "[]")
            w.writerow([
                r["original_name"] or "",
                r["title"] or "",
                r["description"] or "",
                ", ".join(kws),
                r["category"] or "",
            ])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"microstockplus_{ts}.csv"
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),  # BOM for Excel
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={fname}"}
        )
    finally:
        await db.close()


# -- Collections endpoints ------------------------------------------------------
@app.get("/api/v1/collections")
async def list_collections():
    db = await get_db()
    try:
        rows = await (await db.execute(
            "SELECT c.*, COUNT(a.id) as asset_count FROM collections c "
            "LEFT JOIN assets a ON a.collection_id=c.id "
            "GROUP BY c.id ORDER BY c.created_at DESC"
        )).fetchall()
        return {"items": [_coll_dict(r) for r in rows]}
    finally:
        await db.close()


@app.post("/api/v1/collections", status_code=201)
async def create_collection(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Collection name is required")
    coll_id = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO collections (id, name, brief) VALUES (?, ?, ?)",
            (coll_id, name, body.get("brief", ""))
        )
        await db.commit()
        row = await (await db.execute("SELECT * FROM collections WHERE id=?", (coll_id,))).fetchone()
        return _coll_dict(row)
    finally:
        await db.close()


@app.get("/api/v1/collections/{coll_id}")
async def get_collection(coll_id: str):
    db = await get_db()
    try:
        row = await (await db.execute("SELECT * FROM collections WHERE id=?", (coll_id,))).fetchone()
        if not row: raise HTTPException(404, "Collection not found")
        return _coll_dict(row)
    finally:
        await db.close()


@app.put("/api/v1/collections/{coll_id}")
async def update_collection(coll_id: str, body: dict):
    db = await get_db()
    try:
        row = await (await db.execute("SELECT id FROM collections WHERE id=?", (coll_id,))).fetchone()
        if not row: raise HTTPException(404, "Collection not found")
        updates = {}
        if "name"  in body: updates["name"]  = body["name"]
        if "brief" in body: updates["brief"] = body["brief"]
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            await db.execute(
                f"UPDATE collections SET {set_clause}, updated_at=datetime('now') WHERE id=?",
                list(updates.values()) + [coll_id]
            )
            await db.commit()
        row = await (await db.execute("SELECT * FROM collections WHERE id=?", (coll_id,))).fetchone()
        return _coll_dict(row)
    finally:
        await db.close()


@app.delete("/api/v1/collections/{coll_id}", status_code=204)
async def delete_collection(coll_id: str):
    db = await get_db()
    try:
        row = await (await db.execute("SELECT id FROM collections WHERE id=?", (coll_id,))).fetchone()
        if not row: raise HTTPException(404, "Collection not found")
        # Remove collection reference from assets (don't delete assets)
        await db.execute("UPDATE assets SET collection_id=NULL WHERE collection_id=?", (coll_id,))
        await db.execute("DELETE FROM collections WHERE id=?", (coll_id,))
        await db.commit()
    finally:
        await db.close()


def _coll_dict(row):
    d = {
        "id": row["id"], "name": row["name"], "brief": row["brief"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }
    try:
        d["asset_count"] = row["asset_count"]
    except Exception:
        d["asset_count"] = 0
    return d


# -- Health & static ------------------------------------------------------------
@app.get("/api/v1/uploads/{filename}")
async def serve_upload(filename: str):
    """Serve uploaded images for frontend thumbnails."""
    # filename is like <asset_id>.<ext>  -  look up by asset_id prefix
    # Try direct filename first
    direct = UPLOAD_DIR / filename
    if direct.exists():
        return FileResponse(str(direct))
    # Try matching asset_id (prefix before dot)
    asset_id = filename.split('.')[0]
    for p in UPLOAD_DIR.iterdir():
        if p.stem == asset_id:
            return FileResponse(str(p))
    raise HTTPException(404, "File not found")


@app.get("/health")
async def health():
    return {
        "status": "ok", "version": "3.0.0",
        "ai_provider": _provider,
        "ai_model": (AI_MODEL or (_oa_model if OPENAI_API_KEY else "gemini-2.0-flash")),
        "ai_configured": bool(OPENAI_API_KEY or GOOGLE_API_KEY),
    }

@app.get("/", include_in_schema=False)
async def serve_index():
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "StockMeta AI running. Frontend not found."}

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

@app.on_event("startup")
async def startup():
    await init_db()
    if OPENAI_API_KEY:
        log.info("OpenAI configured (model: %s)", _oa_model)
    elif GOOGLE_API_KEY:
        log.info("Google Gemini configured (model: %s)", AI_MODEL or "gemini-2.0-flash")
    else:
        log.warning("No AI provider configured!")
    log.info("StockMeta AI v3.0 running on http://0.0.0.0:%s", os.getenv("PORT", "8000"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
