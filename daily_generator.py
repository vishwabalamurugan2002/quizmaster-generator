"""
QuizMaster Pro Daily Generator with Duplicate Detection
- 10 questions per exam x 4 exams = 40/day
- Dedup: word overlap (Jaccard) + Gemini semantic check
- Translates to 11 languages
- SVG for diagram questions
- Auto-moves yesterday daily to main pool
- Rate limit safe: longer delays, model fallback
Schedule: UPSC 6AM, SSC 12PM, Banking 6PM, RRB 12AM (IST)
"""
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
import json, time, re, os, logging, schedule
from datetime import datetime, timedelta

# ── CONFIG (all secrets from environment variables) ──────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SERVICE_ACCOUNT_PATH = os.environ.get("SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")

QUESTIONS_PER_EXAM = 10
GENERATE_EXTRA = 5
DELAY_API = 15
DELAY_BETWEEN_EXAMS = 600
DELAY_TRANSLATE = 10
DUPLICATE_THRESHOLD = 0.85

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("daily_generator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── EXAM TOPICS ───────────────────────────────────────────────
EXAM_CONFIGS = {
    "UPSC": {
        "topics": [
            "Indian Polity & Constitution",
            "Ancient & Medieval History",
            "Modern History",
            "Indian Geography",
            "Indian Economy",
            "Science & Technology",
            "Art & Culture",
            "Environment & Ecology",
            "International Relations",
            "Current Affairs India"
        ],
        "key": "upsc_idx"
    },
    "SSC": {
        "topics": [
            "General Awareness",
            "Indian History",
            "Geography of India",
            "General Science",
            "Indian Economy",
            "Current Affairs",
            "Indian Polity",
            "Computer & Technology"
        ],
        "key": "ssc_idx"
    },
    "BANKING": {
        "topics": [
            "Banking & Financial Awareness",
            "Indian Economy & RBI",
            "Financial Markets",
            "Current Affairs Banking",
            "Insurance & NBFC",
            "Budget & Economic Survey",
            "International Finance",
            "Digital Banking"
        ],
        "key": "banking_idx"
    },
    "RRB": {
        "topics": [
            "General Science Physics",
            "General Science Chemistry",
            "General Science Biology",
            "Indian Railways GK",
            "General Awareness",
            "Current Affairs",
            "Mathematics",
            "Science & Technology"
        ],
        "key": "rrb_idx"
    }
}

LANGUAGES = {
    "hi": "Hindi", "bn": "Bengali", "te": "Telugu",
    "ta": "Tamil", "mr": "Marathi", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "or": "Odia",
    "pa": "Punjabi", "ur": "Urdu"
}

# ── FIREBASE INIT ─────────────────────────────────────────────
def init_firebase():
    if not firebase_admin._apps:
        creds_b64 = os.environ.get("FIREBASE_CREDENTIALS_B64")
        if creds_b64:
            import base64
            cred_dict = json.loads(base64.b64decode(creds_b64).decode("utf-8"))
            cred = credentials.Certificate(cred_dict)
        else:
            creds_json = os.environ.get("FIREBASE_CREDENTIALS")
            if creds_json:
                cred_dict = json.loads(creds_json)
                cred = credentials.Certificate(cred_dict)
            else:
                cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()

# ── GEMINI CLIENT ─────────────────────────────────────────────
def get_client():
    api_key = GEMINI_API_KEY
    if not api_key:
        raise Exception("GEMINI_API_KEY environment variable not set!")
    return genai.Client(api_key=api_key)

def gemini_call(prompt, retries=4):
    client = get_client()
    # Minimum delay between ALL calls to respect rate limits
    time.sleep(4)

    for model in GEMINI_MODELS:
        for attempt in range(retries):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                log.info(f"API OK ({model})")
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err:
                    m = re.search(r"retry in (\d+)", err)
                    wait = int(m.group(1)) + 15 if m else 90 * (attempt + 1)
                    log.warning(f"Rate limited. Waiting {wait}s on {model}")
                    time.sleep(wait)
                elif "503" in err or "500" in err:
                    wait = 45 * (attempt + 1)
                    log.warning(f"Server error on {model}. Waiting {wait}s")
                    time.sleep(wait)
                elif "404" in err or "403" in err:
                    log.warning(f"Model {model} unavailable, trying next")
                    break
                else:
                    log.error(f"Unexpected error on {model}: {e}")
                    raise e
    raise Exception("All Gemini models failed after all retries")

# ── JSON HELPER ───────────────────────────────────────────────
def clean_json(text):
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    s, e = text.find("["), text.rfind("]") + 1
    if s >= 0 and e > s:
        return text[s:e]
    s, e = text.find("{"), text.rfind("}") + 1
    if s >= 0 and e > s:
        return text[s:e]
    return text

# ── DUPLICATE DETECTION ───────────────────────────────────────
def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower().strip()))

def jaccard(t1, t2):
    w1 = set(normalize(t1).split())
    w2 = set(normalize(t2).split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def load_existing(db, exam_type, max_load=500):
    log.info(f"Loading existing {exam_type} for dedup...")
    existing = []
    try:
        for doc in db.collection(f"examQuestions/{exam_type}/questions").limit(max_load).get():
            d = doc.to_dict()
            if d.get("question_en"):
                existing.append(d["question_en"])
    except Exception as e:
        log.error(f"Load failed: {e}")
    log.info(f"Loaded {len(existing)} existing")
    return existing

def semantic_dup_check(new_q, sample):
    if not sample:
        return False
    try:
        r = gemini_call(
            f'Is this exam question a duplicate of any in the list (same concept, same answer)?\n'
            f'New: "{new_q}"\n'
            f'List: {json.dumps(sample[:12], indent=2)}\n'
            f'Answer ONLY: DUPLICATE or UNIQUE'
        )
        time.sleep(3)
        return "DUPLICATE" in r.upper()
    except:
        return False

def filter_duplicates(new_qs, existing):
    unique, skipped, pool = [], 0, list(existing)
    for q in new_qs:
        text = q.get("question_en", "")
        max_score = max((jaccard(text, ex) for ex in pool), default=0)

        if max_score >= DUPLICATE_THRESHOLD:
            log.info(f"  DUP ({max_score:.2f}): {text[:50]}")
            skipped += 1
            continue

        if max_score >= 0.45:
            log.info(f"  Borderline ({max_score:.2f}) semantic check...")
            if semantic_dup_check(text, pool[:12]):
                log.info(f"  SEMANTIC DUP: {text[:50]}")
                skipped += 1
                continue

        unique.append(q)
        pool.append(text)

    log.info(f"Dedup: {len(unique)} unique, {skipped} removed")
    return unique

# ── TOPIC ROTATION ────────────────────────────────────────────
def get_topic(db, exam_type):
    topics = EXAM_CONFIGS[exam_type]["topics"]
    key = EXAM_CONFIGS[exam_type]["key"]
    ref = db.collection("appConfig").document("topicRotation")
    snap = ref.get()
    idx = snap.to_dict().get(key, 0) if snap.exists else 0
    topic = topics[idx % len(topics)]
    ref.set({key: (idx + 1) % len(topics)}, merge=True)
    return topic

# ── QUESTION GENERATION ───────────────────────────────────────
def generate_questions(exam_type, topic, count, existing_sample=None):
    n = count + GENERATE_EXTRA
    avoid = ""
    if existing_sample:
        avoid = f"\nDo NOT generate questions similar to these existing ones:\n{json.dumps(existing_sample[-8:])}"

    prompt = f"""You are an expert Indian competitive exam question setter.
Generate {n} MCQ questions for {exam_type} exam on topic: {topic}
{avoid}

Include mix:
- 8 regular text questions (different sub-topics, real NCERT facts)
- 2 questions with a simple data table (describe table in question text)
- 1 question with a diagram (write [DIAGRAM: description] in question text)

Output ONLY valid JSON array:
[{{
  "question_en": "full question text",
  "options_en": ["A", "B", "C", "D"],
  "correct": 0,
  "explanation_en": "detailed explanation with facts",
  "category": "{topic}",
  "difficulty": "easy",
  "year": null,
  "exam_tags": ["{exam_type}"],
  "has_image": false,
  "image_description": null
}}]

For diagram questions: set has_image to true and fill image_description."""

    try:
        text = clean_json(gemini_call(prompt))
        qs = json.loads(text)
        valid = [
            q for q in qs
            if q.get("question_en")
            and len(q.get("options_en", [])) == 4
            and isinstance(q.get("correct"), int)
            and 0 <= q["correct"] <= 3
        ]
        log.info(f"Generated {len(valid)} valid questions")
        return valid
    except Exception as e:
        log.error(f"Generation failed: {e}")
        return []

# ── SVG GENERATION ────────────────────────────────────────────
def generate_svg(desc):
    try:
        prompt = (
            f"Create a simple SVG educational diagram for Indian exam.\n"
            f"Description: {desc}\n"
            f"Rules: viewBox=\"0 0 400 300\", simple shapes only (rect, circle, line, text), "
            f"black strokes, white fills, clear labels 12-14px, exam-style clean diagram.\n"
            f"Output ONLY the SVG code starting with <svg"
        )
        svg = gemini_call(prompt)
        s = svg.find("<svg")
        e = svg.rfind("</svg>") + 6
        if s >= 0 and e > s:
            return svg[s:e]
    except Exception as e:
        log.error(f"SVG failed: {e}")
    return None

# ── TRANSLATION ───────────────────────────────────────────────
def translate(q_en, opts_en, exp_en):
    lang_list = ", ".join([f"{v}({k})" for k, v in LANGUAGES.items()])
    prompt = (
        f"Translate this Indian exam MCQ to: {lang_list}\n\n"
        f"Rules:\n"
        f"- Keep unchanged: Gandhi, Ambedkar, Nehru, Delhi, Mumbai, Lok Sabha, Rajya Sabha, "
        f"RBI, article numbers, act names, years, exam terms\n"
        f"- Use native script for each language\n"
        f"- Return ONLY valid JSON\n\n"
        f"Question: {q_en}\n"
        f"Options: {json.dumps(opts_en)}\n"
        f"Explanation: {exp_en}\n\n"
        f'Output ONLY JSON: {{"hi":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"bn":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"te":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"ta":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"mr":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"gu":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"kn":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"ml":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"or":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"pa":{{"question":"","options":["","","",""],"explanation":""}},'
        f'"ur":{{"question":"","options":["","","",""],"explanation":""}}}}'
    )
    try:
        return json.loads(clean_json(gemini_call(prompt)))
    except Exception as e:
        log.error(f"Translation failed: {e}")
        return {}

# ── BUILD FIRESTORE DOCUMENT ──────────────────────────────────
def build_doc(q, translations, svg=None):
    doc = {
        "question_en": q["question_en"],
        "options_en": q["options_en"],
        "correct": q["correct"],
        "explanation_en": q["explanation_en"],
        "category": q.get("category"),
        "difficulty": q.get("difficulty", "medium"),
        "year": q.get("year"),
        "exam_tags": q.get("exam_tags", []),
        "source": "DAILY_AI",
        "has_image": q.get("has_image", False),
        "image_svg": svg,
        "image_description": q.get("image_description"),
        "createdAt": firestore.SERVER_TIMESTAMP,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    for lang, trans in translations.items():
        if isinstance(trans, dict):
            doc[f"question_{lang}"] = trans.get("question", q["question_en"])
            doc[f"options_{lang}"] = trans.get("options", q["options_en"])
            doc[f"explanation_{lang}"] = trans.get("explanation", q["explanation_en"])
    return doc

# ── UPLOAD DAILY QUESTIONS ────────────────────────────────────
def upload_daily(db, exam_type, docs):
    today = datetime.now().strftime("%Y-%m-%d")
    doc_id = f"{today}_{exam_type}"
    ref = db.collection("dailyQuestions").document(doc_id).collection("questions")

    batch = db.batch()
    ids = []
    for d in docs:
        dr = ref.document()
        batch.set(dr, d)
        ids.append(dr.id)
    batch.commit()

    db.collection("dailyQuestions").document(doc_id).set({
        "date": today,
        "exam_type": exam_type,
        "question_count": len(docs),
        "question_ids": ids,
        "uploaded_at": firestore.SERVER_TIMESTAMP,
        "moved_to_pool": False
    }, merge=True)

    log.info(f"Uploaded {len(docs)} → dailyQuestions/{doc_id}")

# ── MOVE YESTERDAY TO MAIN POOL ───────────────────────────────
def move_to_pool(db, exam_type):
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    doc_id = f"{yesterday}_{exam_type}"
    meta = db.collection("dailyQuestions").document(doc_id)
    snap = meta.get()

    if not snap.exists or snap.to_dict().get("moved_to_pool"):
        return 0

    pool = db.collection("examQuestions").document(exam_type).collection("questions")
    batch = db.batch()
    count = 0

    for q in meta.collection("questions").get():
        data = q.to_dict()
        data["source"] = "DAILY_TO_POOL"
        data["moved_date"] = datetime.now().strftime("%Y-%m-%d")
        batch.set(pool.document(), data)
        count += 1
        if count % 499 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()
    meta.update({"moved_to_pool": True})
    log.info(f"Moved {count} questions to main pool: {exam_type}")
    return count

# ── MAIN JOB ──────────────────────────────────────────────────
def run_exam_job(exam_type):
    log.info(f"\n{'='*55}")
    log.info(f"START: {exam_type} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    try:
        db = init_firebase()

        moved = move_to_pool(db, exam_type)
        log.info(f"Moved {moved} to main pool")

        topic = get_topic(db, exam_type)
        log.info(f"Topic: {topic}")

        existing = load_existing(db, exam_type)

        raw = generate_questions(exam_type, topic, QUESTIONS_PER_EXAM, existing)
        if not raw:
            log.error("No questions generated")
            return

        time.sleep(DELAY_API)

        unique = filter_duplicates(raw, existing)
        if not unique:
            log.error("All questions were duplicates!")
            return

        final = unique[:QUESTIONS_PER_EXAM]
        log.info(f"Processing {len(final)} unique questions")

        # Save backup
        os.makedirs("backups", exist_ok=True)
        backup_file = f"backups/{exam_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        log.info(f"Backup saved: {backup_file}")

        # Process each question
        docs = []
        for i, q in enumerate(final):
            log.info(f"Q{i+1}/{len(final)}: {q['question_en'][:45]}")

            # Generate SVG for diagram questions
            svg = None
            if q.get("has_image") and q.get("image_description"):
                log.info(f"  Generating SVG...")
                svg = generate_svg(q["image_description"])
                time.sleep(DELAY_API)

            # Translate to 11 languages
            trans = translate(q["question_en"], q["options_en"], q["explanation_en"])
            time.sleep(DELAY_TRANSLATE)

            docs.append(build_doc(q, trans, svg))
            log.info(f"  Translated: {len(trans)} languages")

        # Upload to Firestore
        upload_daily(db, exam_type, docs)

        log.info(f"DONE: {exam_type}")
        log.info(f"  Questions uploaded: {len(docs)}")
        log.info(f"  Duplicates removed: {len(raw) - len(unique)}")
        log.info(f"  Languages: 11")

    except Exception as e:
        log.error(f"FAILED {exam_type}: {e}")
        import traceback
        log.error(traceback.format_exc())

def run_all():
    exams = ["UPSC", "SSC", "BANKING", "RRB"]
    for i, exam in enumerate(exams):
        run_exam_job(exam)
        if i < len(exams) - 1:
            log.info(f"Waiting {DELAY_BETWEEN_EXAMS}s before next exam...")
            time.sleep(DELAY_BETWEEN_EXAMS)
    log.info("All jobs done!")

# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "now":
            run_all()
        elif arg in ["UPSC", "SSC", "BANKING", "RRB"]:
            run_exam_job(arg)
        elif arg == "move":
            db = init_firebase()
            for e in ["UPSC", "SSC", "BANKING", "RRB"]:
                move_to_pool(db, e)
        elif arg == "test_dedup":
            db = init_firebase()
            exam = sys.argv[2] if len(sys.argv) > 2 else "UPSC"
            ex = load_existing(db, exam)
            if ex:
                result = filter_duplicates([{"question_en": ex[0]}], ex[1:])
                log.info(f"Test: {len(result)} unique (expected 0 — same question)")
        sys.exit(0)

    # Scheduled 24/7 mode
    # UTC times (IST = UTC + 5:30)
    schedule.every().day.at("00:30").do(run_exam_job, "UPSC")     # 6:00 AM IST
    schedule.every().day.at("06:30").do(run_exam_job, "SSC")      # 12:00 PM IST
    schedule.every().day.at("12:30").do(run_exam_job, "BANKING")  # 6:00 PM IST
    schedule.every().day.at("18:30").do(run_exam_job, "RRB")      # 12:00 AM IST

    log.info("QuizMaster Pro Daily Generator Started")
    log.info("Schedule (IST): UPSC=6AM | SSC=12PM | Banking=6PM | RRB=12AM")
    log.info("Running UPSC now as initial test...")

    # Only run UPSC once on startup as test
    run_exam_job("UPSC")

    log.info("Scheduler running 24/7. Waiting for next scheduled time...")
    while True:
        schedule.run_pending()
        time.sleep(60)
