"""
server.py  Ã¢ÂÂ  StockMeta AI  Ã¢ÂÂ  Standalone server (no Docker required)
----------------------------------------------------------------------
Database  : SQLite (built-in, zero setup)
AI        : Google Gemini 2.0 Flash (free tier)
Background: threads (built-in)
Requires  : fastapi  uvicorn  google-generativeai  pillow  python-multipart  aiosqlite
"""

import asyncio
import csv
import io
import json
import logging
import mimetypes
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        created_at    TEXT DEFAULT (datetime('now')),
        updated_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS metadata_outputs (
        id              TEXT PRIMARY KEY,
        asset_id        TEXT NOT NULL,
        platform        TEXT NOT NULL,
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
    """)
    await db.commit()
    await db.close()

ADOBE_PROMPT = """You are an expert Adobe Stock metadata specialist. Analyze this image carefully and generate optimized, accurate metadata.

ADOBE STOCK RULES:
- Title: Title Case, descriptive noun phrase, 5-10 words, 5-80 characters. NO keyword stuffing or comma-separated lists.
- Keywords: 30-45 keywords ranked by relevance (most important first). Avoid banned standalone words: image, photo, stock, concept, background.
- Description: 1-2 sentences describing the scene, subject, mood, and visual elements.
- Category: Choose ONE from: Abstract, Animals/Wildlife, Arts, Backgrounds/Textures, Beauty/Fashion, Buildings/Landmarks, Business/Finance, Education, Food/Drink, Healthcare/Medical, Holidays, Industrial, Nature, Objects, Parks/Outdoor, People, Science, Sports/Recreation, Technology, Transportation, Travel

Respond ONLY with this exact JSON (no markdown, no explanation):
{
  "title": "...",
  "description": "...",
  "keywords": ["kw1", "kw2", "kw3"],
  "category": "..."
}"""

SHUTTERSTOCK_PROMPT = """You are an expert Shutterstock metadata specialist. Analyze this image carefully and generate optimized, accurate metadata.

SHUTTERSTOCK RULES:
- Title (Description field): Complete sentence with subject + verb, Sentence case, minimum 5 words. NO keyword lists, NO comma-separated nouns.
- Keywords: 25-40 keywords ranked by relevance (top 7 most important for search). No filler words.
- Category: Standard Shutterstock category

Respond ONLY with this exact JSON (no markdown, no explanation):
{
  "title": "...",
  "description": "...",
  "keywords": ["kw1", "kw2", "kw3"],
  "category": "..."
}"""

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

        if not OPENAI_API_KEY and not GOOGLE_API_KEY:
            raise ValueError("Neither OPENAI_API_KEY nor GOOGLE_API_KEY is set")

        from PIL import Image as PILImage

        if OPENAI_API_KEY:
            import openai as _openai
            _client = _openai.OpenAI(api_key=OPENAI_API_KEY)
            _model = AI_MODEL or "gpt-4o"
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            _gemini = genai.GenerativeModel(AI_MODEL or "gemini-2.0-flash")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        img = PILImage.open(str(path)).convert("RGB")

        hint_text = ""
        if hints:
            parts = []
            if hints.get("location"):            parts.append(f"Location: {hints['location']}")
            if hints.get("shoot_type"):          parts.append(f"Shoot type: {hints['shoot_type']}")
            if hints.get("people_count") is not None: parts.append(f"People count: {hints['people_count']}")
            if hints.get("intent"):              parts.append(f"Usage intent: {hints['intent']}")
            if parts:
                hint_text = "\n\nPhotographer context:\n" + "\n".join(parts)

        def call_ai(prompt: str) -> dict:
            if OPENAI_API_KEY:
                import base64 as _b64, io as _io
                buf = _io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                img_b64 = _b64.b64encode(buf.getvalue()).decode()
                rsp = _client.chat.completions.create(
                    model=_model,
                    messages=[{"role":"user","content":[
                        {"type":"text","text":prompt+hint_text},
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
                    ]}],
                    max_tokens=1024,
                )
                raw = rsp.choices[0].message.content.strip()
            else:
                response = _gemini.generate_content([prompt + hint_text, img])
                raw = response.text.strip()
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())

        adobe_data = call_ai(ADOBE_PROMPT)
        ss_data    = call_ai(SHUTTERSTOCK_PROMPT)

        await db.execute("UPDATE metadata_outputs SET is_active=0 WHERE asset_id=?", (asset_id,))

        await db.execute(
            "INSERT INTO metadata_outputs (id, asset_id, platform, title, description, keywords, category) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), asset_id, "adobe",
             adobe_data.get("title", ""), adobe_data.get("description", ""),
             json.dumps(adobe_data.get("keywords", [])), adobe_data.get("category", ""))
        )
        await db.execute(
            "INSERT INTO metadata_outputs (id, asset_id, platform, title, description, keywords, category) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), asset_id, "shutterstock",
             ss_data.get("title", ""), ss_data.get("description", ""),
             json.dumps(ss_data.get("keywords", [])), ss_data.get("category", ""))
        )
        await db.execute("UPDATE assets SET status='complete', updated_at=datetime('now') WHERE id=?", (asset_id,))
        await db.commit()
        log.info("Asset %s complete", asset_id)
    except Exception as e:
        log.error("Asset %s failed: %s", asset_id, e)
        await db.execute("UPDATE assets SET status='error', error_msg=?, updated_at=datetime('now') WHERE id=?", (str(e), asset_id))
        await db.commit()
    finally:
        await db.close()

app = FastAPI(title="StockMeta AI", version="2.0.0", docs_url="/docs")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi"}

@app.post("/api/v1/assets/upload", status_code=202)
async def upload_asset(file: UploadFile = File(...), hints: Optional[str] = Form(None)):
    ext = Path(file.filename or "file").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {ext}")
    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 500MB)")

    asset_id = str(uuid.uuid4())
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

    hints_dict = None
    if hints:
        try:
            hints_dict = json.loads(hints)
        except Exception:
            pass

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO assets (id,filename,original_name,asset_type,mime_type,file_size,width,height,storage_path,status,hints) VALUES (?,?,?,?,?,?,?,?,?,'ingested',?)",
            (asset_id, safe_name, file.filename, asset_type, mime, len(content), width, height, store_path, json.dumps(hints_dict) if hints_dict else None)
        )
        await db.commit()
    finally:
        await db.close()

    t = threading.Thread(target=analyze_image_sync, args=(asset_id, store_path, hints_dict), daemon=True)
    t.start()

    return {"asset_id": asset_id, "filename": file.filename, "asset_type": asset_type,
            "status": "ingested", "message": "File accepted. AI generating metadata...",
            "estimated_processing_seconds": 20}


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
async def list_assets(page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=200),
                      status: Optional[str] = Query(None), asset_type: Optional[str] = Query(None)):
    db = await get_db()
    try:
        where, params = "WHERE 1=1", []
        if status:     where += " AND status=?"; params.append(status)
        if asset_type: where += " AND asset_type=?"; params.append(asset_type)
        total = (await (await db.execute(f"SELECT COUNT(*) FROM assets {where}", params)).fetchone())[0]
        rows  = await (await db.execute(f"SELECT * FROM assets {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                                         params + [page_size, (page-1)*page_size])).fetchall()
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
    return {"asset_id": row["id"], "filename": row["original_name"], "asset_type": row["asset_type"],
            "status": row["status"], "width": row["width"], "height": row["height"],
            "duration_secs": row["duration"], "file_size_bytes": row["file_size"],
            "created_at": row["created_at"], "updated_at": row["updated_at"], "error_message": row["error_msg"]}


@app.get("/api/v1/metadata/{asset_id}")
async def get_metadata(asset_id: str):
    db = await get_db()
    try:
        asset = await (await db.execute("SELECT * FROM assets WHERE id=?", (asset_id,))).fetchone()
        if not asset: raise HTTPException(404, "Asset not found")
        if asset["status"] != "complete": raise HTTPException(409, f"Status is '{asset['status']}'")
        rows = await (await db.execute("SELECT * FROM metadata_outputs WHERE asset_id=? AND is_active=1", (asset_id,))).fetchall()
        adobe = next((r for r in rows if r["platform"] == "adobe"), None)
        ss    = next((r for r in rows if r["platform"] == "shutterstock"), None)
        def fmt(r):
            if not r: return None
            kws = json.loads(r["keywords"] or "[]")
            return {"title": r["title"], "description": r["description"],
                    "keywords_ranked": kws, "top10_keywords": kws[:10],
                    "category_primary": r["category"], "version": r["version"]}
        return {"asset_id": asset_id, "asset_type": asset["asset_type"], "adobe": fmt(adobe), "shutterstock": fmt(ss)}
    finally:
        await db.close()


@app.put("/api/v1/metadata/{asset_id}/{platform}")
async def edit_metadata(asset_id: str, platform: str, body: dict):
    if platform not in ("adobe", "shutterstock"): raise HTTPException(400, "Invalid platform")
    db = await get_db()
    try:
        row = await (await db.execute("SELECT id FROM metadata_outputs WHERE asset_id=? AND platform=? AND is_active=1", (asset_id, platform))).fetchone()
        if not row: raise HTTPException(404, "Metadata not found")
        updates = {}
        if "title"           in body: updates["title"]       = body["title"]
        if "description"     in body: updates["description"] = body["description"]
        if "keywords_ranked" in body: updates["keywords"]    = json.dumps(body["keywords_ranked"])
        if "category_primary"in body: updates["category"]    = body["category_primary"]
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            await db.execute(f"UPDATE metadata_outputs SET {set_clause}, manually_edited=1 WHERE id=?",
                             list(updates.values()) + [row["id"]])
            await db.commit()
        return {"status": "updated", "asset_id": asset_id, "platform": platform}
    finally:
        await db.close()


@app.post("/api/v1/metadata/{asset_id}/regenerate", status_code=202)
async def regenerate_metadata(asset_id: str):
    db = await get_db()
    try:
        asset = await (await db.execute("SELECT * FROM assets WHERE id=?", (asset_id,))).fetchone()
        if not asset: raise HTTPException(404, "Asset not found")
        await db.execute("UPDATE metadata_outputs SET is_active=0 WHERE asset_id=?", (asset_id,))
        await db.execute("UPDATE assets SET status='ingested', updated_at=datetime('now') WHERE id=?", (asset_id,))
        await db.commit()
    finally:
        await db.close()
    t = threading.Thread(target=analyze_image_sync,
                         args=(asset_id, str(UPLOAD_DIR / asset["filename"]), json.loads(asset["hints"] or "{}")),
                         daemon=True)
    t.start()
    return {"status": "queued", "message": "Regeneration started"}


@app.post("/api/v1/export/csv")
async def export_csv(platform: str = Query(..., pattern="^(adobe|shutterstock)$"),
                     asset_ids: Optional[list[str]] = Query(None)):
    db = await get_db()
    try:
        where, params = "WHERE a.status='complete'", [platform]
        if asset_ids:
            ph = ",".join("?" * len(asset_ids))
            where += f" AND a.id IN ({ph})"
            params.extend(asset_ids)
        rows = await (await db.execute(
            f"SELECT a.*, m.title, m.description, m.keywords, m.category "
            f"FROM assets a JOIN metadata_outputs m ON m.asset_id=a.id AND m.platform=? AND m.is_active=1 "
            f"{where} ORDER BY a.created_at DESC", params)).fetchall()
        if not rows: raise HTTPException(404, "No completed assets found")
        output = io.StringIO()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if platform == "adobe":
            w = csv.DictWriter(output, fieldnames=["Filename","Title","Keywords","Category","Description"], lineterminator="\n")
            w.writeheader()
            for r in rows:
                kws = json.loads(r["keywords"] or "[]")
                w.writerow({"Filename": r["original_name"], "Title": r["title"] or "",
                            "Keywords": "; ".join(kws), "Category": r["category"] or "",
                            "Description": r["description"] or ""})
            fname = f"adobe_stock_{ts}.csv"
        else:
            w = csv.DictWriter(output, fieldnames=["Filename","Description","Keywords","Categories","Editorial"], lineterminator="\n")
            w.writeheader()
            for r in rows:
                kws = json.loads(r["keywords"] or "[]")
                w.writerow({"Filename": r["original_name"], "Description": r["title"] or "",
                            "Keywords": ", ".join(kws), "Categories": r["category"] or "", "Editorial": "no"})
            fname = f"shutterstock_{ts}.csv"
        output.seek(0)
        return StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8")), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename={fname}"})
    finally:
        await db.close()


@app.get("/health")
async def health():
    provider = "OpenAI" if OPENAI_API_KEY else "Google Gemini"
    model_name = AI_MODEL or ("gpt-4o" if OPENAI_API_KEY else "gemini-2.0-flash")
    return {"status": "ok", "version": "2.0.0", "ai_provider": provider,
            "ai_model": model_name, "ai_configured": bool(OPENAI_API_KEY or GOOGLE_API_KEY)}

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
    if not GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not set!")
    else:
        log.info("Google Gemini configured (model: %s)", AI_MODEL)
    log.info("StockMeta AI running on http://0.0.0.0:%s", os.getenv("PORT", "8000"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
