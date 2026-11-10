#!/usr/bin/env python3
"""clip-factory orchestrator.

Downloads a YouTube VOD, finds viral moments with Gemini, transcribes each
clip with Whisper for word-level captions, and renders a vertical short.
"""
import argparse, sys, os, tempfile, statistics, re, json, subprocess, random, uuid
from pathlib import Path

import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None  # face tracking disabled, center-crop fallback is used

import whisper
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi


# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
EXPORT_DIR = BASE_DIR / "exports"
BG_VIDEO = BASE_DIR / "assets" / "background.mp4"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
WHISPER_MODEL_NAME = "base"

MANIFEST_PATH = EXPORT_DIR / "manifest.json"
DEFAULT_WATERMARK = BASE_DIR / "watermarks" / "logo.png"
BLOCKLIST = re.compile(
    r"\b(gambl\w*|casino\w*|betting|sportsbook|bookmaker|parlay|wager\w*|slot machines?|"
    r"poker|blackjack|roulette|draftkings|fanduel|porn\w*|nsfw|nude\w*|naked|sex|sexual\w*|"
    r"sexy|onlyfans|xxx|erotic\w*|escort\w*)\b", re.I)


def is_blocked(text: str) -> bool:
    return bool(BLOCKLIST.search(text or ""))


# ---------------------------------------------------------------- helpers
def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"Could not extract video id from {url}")
    return m.group(1)


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:40] or "clip"


def probe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip())


# ---------------------------------------------------------------- transcript
def fetch_transcript(video_id, url) -> list[dict]:
    try:
        try:
            raw = YouTubeTranscriptApi.get_transcript(video_id)
        except AttributeError:
            raw = [{"start": s.start, "text": s.text}
                   for s in YouTubeTranscriptApi().fetch(video_id)]
        return [{"start": float(e["start"]), "text": e["text"]} for e in raw]
    except Exception as e:
        print(f"  ⚠️  Transcript API failed ({e}); falling back to yt-dlp")
        return fetch_transcript_ytdlp(video_id, url)


def fetch_transcript_ytdlp(video_id, url):
    outbase = TEMP_DIR / f"subs_{video_id}.%(ext)s"
    subprocess.run(
        ["yt-dlp", "--write-auto-sub", "--sub-langs", "en", "--sub-format", "vtt",
         "--skip-download", "--no-playlist", "-o", str(outbase), url],
        check=True)
    vtt = next(TEMP_DIR.glob(f"subs_{video_id}*.vtt"), None)
    if vtt is None:
        raise RuntimeError("No captions available")
    entries = []
    last_text = None
    ts_re = re.compile(r"^(\d+):(\d+):(\d+\.\d+) --> ")
    try:
        start = 0.0
        with vtt.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                m = ts_re.match(stripped)
                if m:
                    h, mm, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    start = h * 3600 + mm * 60 + s
                    continue
                if not stripped:
                    continue
                if stripped.startswith(("WEBVTT", "Kind:", "Language:")):
                    continue
                text = re.sub(r"<[^>]+>", "", stripped).strip()
                if not text:
                    continue
                if text == last_text:
                    continue
                last_text = text
                entries.append({"start": start, "text": text})
    finally:
        vtt.unlink(missing_ok=True)
    return entries


# ---------------------------------------------------------------- Gemini
def get_viral_moments_with_gemini(transcript: list, max_clips: int) -> list:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    formatted_lines = "\n".join(
        f"[{int(e['start'])}s] {e['text']}" for e in transcript)

    prompt = f"""You are an expert short-form video editor picking viral moments.

Transcript (each line begins with its start time in seconds):
{formatted_lines}

Rules:
- Pick moments that are 20-60 seconds long and likely to go viral.
- Prefer story beats, hot takes, confessions, surprising facts and stakes.
- Start on a hook, end on a punchline or reveal.
- STRICT FILTER: never propose clips whose content involves gambling, betting,
  casinos, sportsbooks, adult content, or any prohibited topics.
- Return up to {max_clips + 3} candidates so rejected ones can be replaced.
- Return ONLY a valid JSON array of objects with keys:
  "start" (integer seconds), "end" (integer seconds),
  "hook_title" (short 4-word file slug), "virality_score" (1-100),
  "caption" (curiosity-inducing or debate-baiting caption, followed by exactly 4 niche hashtags),
  "on_screen_hook" (exactly 5 words, punchy text hook shown in the first 3 seconds).
"""

    response = model.generate_content(prompt)
    data = json.loads(response.text)
    if isinstance(data, dict):
        data = data.get("moments", [])

    def score(m):
        try:
            return int(m.get("virality_score", 0))
        except (TypeError, ValueError):
            return 0

    return sorted(data, key=score, reverse=True)


# ---------------------------------------------------------------- yt-dlp
def download_clip_slice(url, start, end, output_path):
    cmd = ["yt-dlp",
           "--download-sections", f"*{start}-{end}",
           "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
           "--no-playlist",
           "--force-overwrites",
           "--merge-output-format", "mp4",
           "-o", str(output_path),
           url]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- framing
def analyze_framing(video_path: Path):
    """Returns (src_w, src_h, cx_norm_or_None). cx_norm is 0..1."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not w or not h:
            raise RuntimeError(f"Cannot read video dimensions: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(fps / 2))  # ~2 samples per second
        faces = []  # (cx_norm, area_norm)
        if mp is not None:
            try:
                with mp.solutions.face_detection.FaceDetection(
                        model_selection=1, min_detection_confidence=0.6) as fd:
                    i = 0
                    while cap.grab():
                        if i % step == 0:
                            ok, frame = cap.retrieve()
                            if ok:
                                res = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                for d in (res.detections or []):
                                    bb = d.location_data.relative_bounding_box
                                    faces.append((bb.xmin + bb.width / 2,
                                                  bb.width * bb.height))
                        i += 1
            except Exception as e:
                print(f"  face tracking failed ({e}); using center crop")
                faces = []
        if len(faces) < 3:
            return w, h, None
        # Cluster into 0.25-wide horizontal bins; the primary speaker is the
        # bin with the largest summed face area.
        bins = {}
        for cx, area in faces:
            bins.setdefault(min(int(cx / 0.25), 3), []).append((cx, area))
        best = max(bins.values(), key=lambda v: sum(a for _, a in v))
        return w, h, statistics.median(cx for cx, _ in best)
    finally:
        cap.release()


def compute_crop(src_w, src_h, out_h, cx_norm):
    """Returns (scaled_w, scaled_h, crop_x, crop_y) for a 1080 x out_h crop."""
    factor = max(1080 / src_w, out_h / src_h) * 1.03  # 3% zoom
    sw = int(round(src_w * factor / 2) * 2)
    sh = int(round(src_h * factor / 2) * 2)
    cx = (cx_norm if cx_norm is not None else 0.5) * sw
    x = int(min(max(cx - 540, 0), sw - 1080))
    y = max(0, (sh - out_h) // 2)
    return sw, sh, x, y


# ---------------------------------------------------------------- captions
def generate_word_level_ass(video_path, ass_path, whisper_model, hook: str = ""):
    result = whisper_model.transcribe(str(video_path), word_timestamps=True)

    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CrayoPop,Arial Black,90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,80,80,320,1
Style: Hook,Arial Black,70,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,8,60,60,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def fmt_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    events = []
    spoken = []
    for segment in result["segments"]:
        if not segment.words:
            continue
        spoken.append(segment.text)
        for w in segment.words:
            word = w["word"].strip()
            if not word:
                continue
            start = w["start"]
            end = w["end"]
            dur = max(0.0, end - start)
            fade = min(80, int(dur * 1000 / 4))
            events.append(
                f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},CrayoPop,,0,0,0,,"
                f"{{\\fad({fade},{fade})}}{word}"
            )
    if hook:
        safe = re.sub(r"[{}\\]", "", hook).upper()
        events.insert(0,
            f"Dialogue: 1,0:00:00.00,0:00:03.00,Hook,,0,0,0,,{{\\fad(150,150)}}{safe}")

    ass_path.write_text(ass_header + "\n".join(events) + "\n", encoding="utf-8")
    return " ".join(spoken)


# ---------------------------------------------------------------- rendering
def render_vertical(raw_clip, ass_path, output_path, duration, mode, watermark):
    print(f"🎬 Rendering {mode} vertical: {output_path}")
    src_w, src_h, cx = analyze_framing(raw_clip)
    split = (mode == "split")
    out_h = 960 if split else 1920
    sw, sh, x, y = compute_crop(src_w, src_h, out_h, cx)
    sat = round(random.uniform(1.02, 1.06), 3)
    con = round(random.uniform(1.01, 1.04), 3)
    ass_f = ass_path.as_posix().replace(":", r"\:")

    inputs = ["-i", str(raw_clip)]
    top = f"[0:v]scale={sw}:{sh},crop=1080:{out_h}:{x}:{y},eq=saturation={sat}:contrast={con}"
    if split:
        bg_dur = probe_duration(BG_VIDEO)
        bg_start = random.uniform(0, max(0.0, bg_dur - duration - 1))
        inputs += ["-ss", f"{bg_start:.3f}", "-t", str(duration), "-i", str(BG_VIDEO)]
        chain = [top + "[top]",
                 "[1:v]scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[bottom]",
                 "[top][bottom]vstack=inputs=2[stacked]"]
        wm_idx = 2
    else:
        chain = [top + "[stacked]"]
        wm_idx = 1

    if watermark:
        inputs += ["-loop", "1", "-t", str(duration), "-i", str(watermark)]
        chain += [f"[stacked]ass='{ass_f}'[subbed]",
                  f"[{wm_idx}:v]scale=180:-1,format=rgba,colorchannelmixer=aa=0.85[wm]",
                  "[subbed][wm]overlay=x=W-w-40:y=80[outv]"]
    else:
        chain += [f"[stacked]ass='{ass_f}'[outv]"]

    cmd = ["ffmpeg", "-y", *inputs,
           "-filter_complex", ";".join(chain),
           "-map", "[outv]", "-map", "0:a?",
           "-map_metadata", "-1", "-map_chapters", "-1",
           "-c:v", "h264_videotoolbox", "-b:v", "6M", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart",
           "-t", str(duration), "-shortest",
           str(output_path)]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- manifest
def append_manifest(entry: dict):
    data = []
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    data.append(entry)
    fd, tmp = tempfile.mkstemp(dir=EXPORT_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST_PATH)  # atomic


# ---------------------------------------------------------------- main
def main(args):
    TEMP_DIR.mkdir(exist_ok=True)
    EXPORT_DIR.mkdir(exist_ok=True)

    if args.mode == "split" and not BG_VIDEO.exists():
        print(f"❌ Background video not found: {BG_VIDEO}")
        sys.exit(1)

    if args.watermark:
        watermark = Path(args.watermark).expanduser()
        if not watermark.exists():
            print(f"❌ Watermark not found: {watermark}")
            sys.exit(1)
    elif DEFAULT_WATERMARK.exists():
        watermark = DEFAULT_WATERMARK
    else:
        watermark = None

    video_id = extract_video_id(args.url)
    print(f"📼 Video id: {video_id}")
    transcript = fetch_transcript(video_id, args.url)
    print(f"📝 Transcript entries: {len(transcript)}")

    moments = get_viral_moments_with_gemini(transcript, args.max_clips)
    print(f"🎯 Moments: {len(moments)}")

    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    done = 0

    for idx, m in enumerate(moments, 1):
        if done >= args.max_clips:
            break
        try:
            start = int(m["start"])
            end = int(m["end"])
        except (KeyError, TypeError, ValueError):
            continue
        duration = end - start
        if not (20 <= duration <= 60):
            print(f"  ⏭️  Clip #{idx} skipped: duration {duration}s out of range")
            continue
        hook = " ".join(str(m.get("on_screen_hook", "")).split()[:5])
        caption = str(m.get("caption", "")).strip()
        window = " ".join(e["text"] for e in transcript
                          if start <= e["start"] <= end)
        if is_blocked(f"{window} {caption} {hook}"):
            print(f"  ⛔ Clip #{idx} skipped: content filter")
            continue

        slug = slugify(str(m.get("hook_title", f"clip_{idx}")))
        tag = uuid.uuid4().hex[:8]
        raw_path = TEMP_DIR / f"raw_{slug}_{tag}.mp4"
        ass_path = TEMP_DIR / f"cap_{slug}_{tag}.ass"
        final_path = EXPORT_DIR / f"{slug}_{tag}.mp4"

        try:
            download_clip_slice(args.url, start, end, raw_path)
            spoken = generate_word_level_ass(raw_path, ass_path, whisper_model, hook)
            if is_blocked(spoken):
                print("  ⛔ Skipped: spoken content failed the content filter")
                continue
            render_vertical(raw_path, ass_path, final_path, duration,
                            args.mode, watermark)
            append_manifest({
                "file": final_path.name, "source_url": args.url,
                "start": start, "end": end, "mode": args.mode,
                "virality_score": m.get("virality_score"),
                "caption": caption, "on_screen_hook": hook, "cta": args.cta,
            })
            done += 1
            print(f"✅ Exported: {final_path}")
        except Exception as e:  # includes CalledProcessError; one bad clip must not kill the run
            print(f"❌ Clip #{idx} failed: {e}")
        finally:
            raw_path.unlink(missing_ok=True)
            ass_path.unlink(missing_ok=True)

    print(f"🏁 Done. Exported {done} clip(s).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="clip-factory orchestrator")
    p.add_argument("--url", required=True, help="YouTube VOD URL")
    p.add_argument("--max-clips", type=int, default=5)
    p.add_argument("--mode", choices=["split", "fullscreen"], default="split")
    p.add_argument("--watermark", default=None, help="Path to transparent PNG (overrides watermarks/logo.png)")
    p.add_argument("--cta", default="", help="Optional required campaign CTA stored in manifest")
    main(p.parse_args())
