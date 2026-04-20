"""
QuizMaster Pro Daily Generator with Duplicate Detection
- 10 questions per exam x 4 exams = 40/day
- Dedup: word overlap (Jaccard) + Gemini semantic check
- Translates to 11 languages
- SVG for diagram questions
- Auto-moves yesterday daily to main pool
Schedule: UPSC 6AM, SSC 12PM, Banking 6PM, RRB 12AM (IST)
"""
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
import json, time, re, os, logging, schedule
from datetime import datetime, timedelta

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDUetTNXjAKBYeYfJsUpu718NRv-bIqQSw")
SERVICE_ACCOUNT_PATH = os.environ.get("SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
QUESTIONS_PER_EXAM = 10
GENERATE_EXTRA = 5
DELAY_API = 8
DELAY_BETWEEN_EXAMS = 300
DELAY_TRANSLATE = 6
DUPLICATE_THRESHOLD = 0.85

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("daily_generator.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

EXAM_CONFIGS = {
    "UPSC": {"topics": ["Indian Polity & Constitution","Ancient & Medieval History",
        "Modern History","Indian Geography","Indian Economy","Science & Technology",
        "Art & Culture","Environment & Ecology","International Relations","Current Affairs India"],
        "key": "upsc_idx"},
    "SSC": {"topics": ["General Awareness","Indian History","Geography of India",
        "General Science","Indian Economy","Current Affairs","Indian Polity","Computer & Technology"],
        "key": "ssc_idx"},
    "BANKING": {"topics": ["Banking & Financial Awareness","Indian Economy & RBI",
        "Financial Markets","Current Affairs Banking","Insurance & NBFC",
        "Budget & Economic Survey","International Finance","Digital Banking"], "key": "banking_idx"},
    "RRB": {"topics": ["General Science Physics","General Science Chemistry",
        "General Science Biology","Indian Railways GK","General Awareness",
        "Current Affairs","Mathematics","Science & Technology"], "key": "rrb_idx"}
}

LANGUAGES = {"hi":"Hindi","bn":"Bengali","te":"Telugu","ta":"Tamil","mr":"Marathi",
    "gu":"Gujarati","kn":"Kannada","ml":"Malayalam","or":"Odia","pa":"Punjabi","ur":"Urdu"}

client = genai.Client(api_key=GEMINI_API_KEY)

def init_firebase():
    if not firebase_admin._apps:
        # Try base64 encoded credentials first
        creds_b64 = os.environ.get("FIREBASE_CREDENTIALS_B64")
        if creds_b64:
            import base64
            cred_dict = json.loads(base64.b64decode(creds_b64).decode('utf-8'))
            cred = credentials.Certificate(cred_dict)
        else:
            # Try raw JSON credentials
            creds_json = os.environ.get("FIREBASE_CREDENTIALS")
            if creds_json:
                cred_dict = json.loads(creds_json)
                cred = credentials.Certificate(cred_dict)
            else:
                # Fall back to file
                cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def gemini_call(prompt, retries=4):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err:
                m = re.search(r"retry in (\d+)", err)
                wait = int(m.group(1)) + 10 if m else 60*(attempt+1)
                log.warning(f"Rate limited {wait}s"); time.sleep(wait)
            elif "503" in err or "500" in err:
                time.sleep(30*(attempt+1))
            else:
                raise e
    raise Exception("Gemini failed")

def clean_json(text):
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    s, e = text.find("["), text.rfind("]")+1
    if s >= 0 and e > s: return text[s:e]
    s, e = text.find("{"), text.rfind("}")+1
    if s >= 0 and e > s: return text[s:e]
    return text

def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower().strip()))

def jaccard(t1, t2):
    w1, w2 = set(normalize(t1).split()), set(normalize(t2).split())
    if not w1 or not w2: return 0.0
    return len(w1 & w2) / len(w1 | w2)

def load_existing(db, exam_type, max_load=500):
    log.info(f"Loading existing {exam_type} for dedup...")
    existing = []
    try:
        for doc in db.collection(f"examQuestions/{exam_type}/questions").limit(max_load).get():
            d = doc.to_dict()
            if d.get("question_en"): existing.append(d["question_en"])
    except Exception as e:
        log.error(f"Load failed: {e}")
    log.info(f"Loaded {len(existing)} existing"); return existing

def semantic_dup_check(new_q, sample):
    if not sample: return False
    try:
        r = gemini_call(f"""Is this exam question a duplicate of any in the list (same concept)?
New: "{new_q}"
List: {json.dumps(sample[:12], indent=2)}
Answer ONLY: DUPLICATE or UNIQUE""")
        time.sleep(3); return "DUPLICATE" in r.upper()
    except: return False

def filter_duplicates(new_qs, existing):
    unique, skipped, pool = [], 0, list(existing)
    for q in new_qs:
        text = q.get("question_en", "")
        max_score = max((jaccard(text, ex) for ex in pool), default=0)
        if max_score >= DUPLICATE_THRESHOLD:
            log.info(f"  DUP ({max_score:.2f}): {text[:50]}"); skipped += 1; continue
        if max_score >= 0.45:
            if semantic_dup_check(text, pool[:12]):
                log.info(f"  SEMANTIC DUP: {text[:50]}"); skipped += 1; continue
        unique.append(q); pool.append(text)
    log.info(f"Dedup: {len(unique)} unique, {skipped} removed"); return unique

def get_topic(db, exam_type):
    topics = EXAM_CONFIGS[exam_type]["topics"]
    key = EXAM_CONFIGS[exam_type]["key"]
    ref = db.collection("appConfig").document("topicRotation")
    snap = ref.get()
    idx = snap.to_dict().get(key, 0) if snap.exists else 0
    topic = topics[idx % len(topics)]
    ref.set({key: (idx+1) % len(topics)}, merge=True)
    return topic

def generate_questions(exam_type, topic, count, existing_sample=None):
    n = count + GENERATE_EXTRA
    avoid = f"\nDo NOT repeat: {json.dumps((existing_sample or [])[-8:])}" if existing_sample else ""
    prompt = f"""Expert Indian exam question setter.
Generate {n} MCQ for {exam_type} on: {topic}{avoid}
Mix: 8 text + 2 table-based + 1 diagram ([DIAGRAM: desc] in question)
Output ONLY JSON array:
[{{"question_en":"...","options_en":["A","B","C","D"],"correct":0,"explanation_en":"...","category":"{topic}","difficulty":"easy","year":null,"exam_tags":["{exam_type}"],"has_image":false,"image_description":null}}]"""
    try:
        qs = json.loads(clean_json(gemini_call(prompt)))
        valid = [q for q in qs if q.get("question_en") and len(q.get("options_en",[]))==4
                 and isinstance(q.get("correct"),int) and 0<=q["correct"]<=3]
        log.info(f"Generated {len(valid)} valid"); return valid
    except Exception as e:
        log.error(f"Generation failed: {e}"); return []

def generate_svg(desc):
    try:
        svg = gemini_call(f"Simple SVG exam diagram. Description: {desc}\nviewBox=\"0 0 400 300\", simple shapes, clear labels. Output ONLY SVG starting with <svg")
        s, e = svg.find("<svg"), svg.rfind("</svg>")+6
        if s >= 0 and e > s: return svg[s:e]
    except Exception as e: log.error(f"SVG failed: {e}")
    return None

def translate(q_en, opts_en, exp_en):
    lang_list = ", ".join([f"{v}({k})" for k,v in LANGUAGES.items()])
    prompt = f"""Translate Indian exam MCQ to: {lang_list}
Keep unchanged: Gandhi, Ambedkar, Delhi, Lok Sabha, RBI, article numbers, act names, years
Q: {q_en}\nOptions: {json.dumps(opts_en)}\nExplanation: {exp_en}
Output ONLY JSON with keys: hi,bn,te,ta,mr,gu,kn,ml,or,pa,ur each having question,options,explanation"""
    try:
        return json.loads(clean_json(gemini_call(prompt)))
    except Exception as e:
        log.error(f"Translation failed: {e}"); return {}

def build_doc(q, translations, svg=None):
    doc = {"question_en":q["question_en"],"options_en":q["options_en"],"correct":q["correct"],
           "explanation_en":q["explanation_en"],"category":q.get("category"),
           "difficulty":q.get("difficulty","medium"),"year":q.get("year"),
           "exam_tags":q.get("exam_tags",[]),"source":"DAILY_AI","has_image":q.get("has_image",False),
           "image_svg":svg,"image_description":q.get("image_description"),
           "createdAt":firestore.SERVER_TIMESTAMP,"date":datetime.now().strftime("%Y-%m-%d")}
    for lang, trans in translations.items():
        if isinstance(trans, dict):
            doc[f"question_{lang}"] = trans.get("question", q["question_en"])
            doc[f"options_{lang}"] = trans.get("options", q["options_en"])
            doc[f"explanation_{lang}"] = trans.get("explanation", q["explanation_en"])
    return doc

def upload_daily(db, exam_type, docs):
    today = datetime.now().strftime("%Y-%m-%d")
    ref = db.collection("dailyQuestions").document(f"{today}_{exam_type}").collection("questions")
    batch, ids = db.batch(), []
    for d in docs:
        dr = ref.document(); batch.set(dr, d); ids.append(dr.id)
    batch.commit()
    db.collection("dailyQuestions").document(f"{today}_{exam_type}").set(
        {"date":today,"exam_type":exam_type,"question_count":len(docs),"question_ids":ids,
         "uploaded_at":firestore.SERVER_TIMESTAMP,"moved_to_pool":False}, merge=True)
    log.info(f"Uploaded {len(docs)} to dailyQuestions/{today}_{exam_type}")

def move_to_pool(db, exam_type):
    yesterday = (datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d")
    meta = db.collection("dailyQuestions").document(f"{yesterday}_{exam_type}")
    snap = meta.get()
    if not snap.exists or snap.to_dict().get("moved_to_pool"): return 0
    pool = db.collection("examQuestions").document(exam_type).collection("questions")
    batch, count = db.batch(), 0
    for q in meta.collection("questions").get():
        data = q.to_dict()
        data["source"] = "DAILY_TO_POOL"
        data["moved_date"] = datetime.now().strftime("%Y-%m-%d")
        batch.set(pool.document(), data); count += 1
        if count % 499 == 0: batch.commit(); batch = db.batch()
    batch.commit(); meta.update({"moved_to_pool":True})
    log.info(f"Moved {count} to pool: {exam_type}"); return count

def run_exam_job(exam_type):
    log.info(f"\n{'='*55}\nSTART: {exam_type} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    try:
        db = init_firebase()
        log.info(f"Moved {move_to_pool(db, exam_type)} to pool")
        topic = get_topic(db, exam_type)
        log.info(f"Topic: {topic}")
        existing = load_existing(db, exam_type)
        raw = generate_questions(exam_type, topic, QUESTIONS_PER_EXAM, existing)
        if not raw: log.error("No questions generated"); return
        time.sleep(DELAY_API)
        unique = filter_duplicates(raw, existing)
        if not unique: log.error("All duplicates!"); return
        final = unique[:QUESTIONS_PER_EXAM]
        os.makedirs("backups", exist_ok=True)
        with open(f"backups/{exam_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.json","w",encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        docs = []
        for i, q in enumerate(final):
            log.info(f"Q{i+1}/{len(final)}: {q['question_en'][:45]}")
            svg = None
            if q.get("has_image") and q.get("image_description"):
                svg = generate_svg(q["image_description"]); time.sleep(DELAY_API)
            trans = translate(q["question_en"], q["options_en"], q["explanation_en"])
            time.sleep(DELAY_TRANSLATE)
            docs.append(build_doc(q, trans, svg))
            log.info(f"  {len(trans)} langs")
        upload_daily(db, exam_type, docs)
        log.info(f"DONE: {exam_type} - {len(docs)} questions, {len(raw)-len(unique)} dups removed")
    except Exception as e:
        log.error(f"FAILED {exam_type}: {e}")
        import traceback; log.error(traceback.format_exc())

def run_all():
    for i, exam in enumerate(["UPSC","SSC","BANKING","RRB"]):
        run_exam_job(exam)
        if i < 3: time.sleep(DELAY_BETWEEN_EXAMS)
    log.info("All jobs done!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "now": run_all()
        elif arg in ["UPSC","SSC","BANKING","RRB"]: run_exam_job(arg)
        elif arg == "move":
            db = init_firebase()
            for e in ["UPSC","SSC","BANKING","RRB"]: move_to_pool(db, e)
        elif arg == "test_dedup":
            db = init_firebase()
            exam = sys.argv[2] if len(sys.argv)>2 else "UPSC"
            ex = load_existing(db, exam)
            if ex:
                result = filter_duplicates([{"question_en": ex[0]}], ex[1:])
                log.info(f"Test result: {len(result)} unique (expected 0)")
        sys.exit(0)
    schedule.every().day.at("00:30").do(run_exam_job, "UPSC")
    schedule.every().day.at("06:30").do(run_exam_job, "SSC")
    schedule.every().day.at("12:30").do(run_exam_job, "BANKING")
    schedule.every().day.at("18:30").do(run_exam_job, "RRB")
    log.info("Running initial job...")
    run_all()
    while True: schedule.run_pending(); time.sleep(60)
