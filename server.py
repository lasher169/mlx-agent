"""
Threaded HTTP gateway server.

Accepts JSON requests from LAN clients. Every request's first step is a
classification: the chat model decides whether the user's text is a chat
message or an image-generation prompt. It is then routed to image
generation or a chat response.

MLX models (chat model_1 + FLUX) are thread-local: they must run on ONE
thread or MLX raises "There is no Stream(cpu, 0) in current thread".
So all MLX work happens on a single atomic worker thread.

Start the server:
    python server.py

Test it (from any LAN PC):
    curl -X POST http://192.168.50.123:8000/ \
        -H "Content-Type: application/json" \
        -d '{"prompt": "a red fox in the snow"}'
"""

import argparse
from datetime import datetime
import queue
import re
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)


def _now_str():
    """Wall-clock timestamp with millisecond precision, for log lines."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# ---- Worker-queue plumbing ------------------------------------------
# The MLX chat model and FLUX both run on ONE atomic worker thread (MLX
# streams are thread-local). Requests arrive in _inbox; the worker classifies
# each one ("image" or "chat") and processes it inline. There is no second
# consumer thread, because that would run MLX off-thread and raise
# "There is no Stream(cpu, 0) in current thread." Routing to image/chat is
# therefore decided on the worker thread itself.
_inbox = queue.Queue()                # items: {"id", "prompt"}
_results = {}                         # id -> {"done": Event, "image", "text", "error"}
_results_lock = threading.Lock()


def _strip_thinking(raw):
    """Drop the model's thinking channel; keep only the final answer.

    The thinking model emits channel blocks like
    '<|channel>thought ...reasoning... <channel|>final answer'. Only the
    text after the LAST channel marker is the actual answer; the reasoning
    may mention 'image'/'chat' and must not influence parsing.
    """
    parts = re.split(r"<\|?channel\|?>", raw)  # tolerate <|channel> / <channel|> / <|channel|>
    if len(parts) == 1:
        return raw.strip()                     # no channel markers: normal output
    final = parts[-1].strip()
    return final if final else raw.strip()     # guard against truncated generation


def _parse_classification(raw):
    """Extract "chat" or "image" from a (possibly thinking) classifier reply.

    Only text after the last channel marker is judged. If the generation was
    truncated at a marker (empty final part), returns None rather than
    scanning the reasoning, which may mention "image".
    """
    parts = re.split(r"<\|?channel\|?>", raw)
    final = (parts[-1] if len(parts) > 1 else raw).strip().lower()
    if not final:
        return None
    words = re.findall(r"\b(chat|image)\b", final)
    return words[-1] if words else None


def _classify(prompt, send):
    """Ask the chat model whether this is a chat or an image prompt.

    Returns "image" or "chat" by asking the model to answer just that."""
    question = (
        f"As a prompt classifier, tell me whether this is a chat prompt or an "
        f"image prompt: '{prompt}'. If it is a chat prompt, respond with exactly "
        f"'chat'. If it is an image prompt, respond with exactly 'image'. If you are unsure, respond with 'unsure'. Do not add any other text or explanation."
    )
    try:
        answer = send(question, model_type="model1") or ""
        #print(f"[Worker] Classifier response: {answer!r}")
        kind = _parse_classification(answer)
        #print(f"[Worker] Parsed classification: {kind!r}")
    except Exception as e:
        #print(f"[Worker] Classification failed (defaulting to chat): {e}")
        kind = None
    return kind or "chat"


def _pipeline_worker():
    """Single atomic thread: load models, then classify and process requests.

    MLX streams are thread-local, so models load + run inference on this one
    thread only. Each request is classified by the chat model, then routed to
    image generation or a chat response - decided here, on the worker thread."""
    from initialise import load_models_mlx, load_models_cuda, generate_image, send

    
    load_models_mlx()
    
    print("\n model loaded, worker thread ready to CURL\n")

    while True:
        job = _inbox.get()  # blocks; classification + processing on one thread
        job_id = job["id"]
        prompt = job["prompt"]
        try:
            # Step 1: route the request via the chat model.
            print(f"[Main processed command]: {prompt}")
            print("\n\n####classify start time =", _now_str())

            classify_t0 = time.perf_counter()
            kind = _classify(prompt, send)
            classify_elapsed = time.perf_counter() - classify_t0

            print(f"[Classfication] → {kind})")
            print(f"####Classify finish time = {_now_str()} (took {classify_elapsed:.2f}s)")

            if kind == "image":
                print(f"[Worker] → image queue: {prompt}")
                print(f"[Worker] Generating image for prompt: {prompt}")

                image_t0 = time.perf_counter()
                image = generate_image(prompt)
                image_elapsed = time.perf_counter() - image_t0
                image.save(path="my_generated_image.png")

                print(f"[Worker] Image done (took {image_elapsed:.2f}s)")
                text, error = None, None
            else:
                print(f"[Worker] → Chat queue: {prompt}")
                print("\n\n####chat start time =", _now_str())

                chat_t0 = time.perf_counter()
                text = send(prompt, model_type="model1")
                chat_elapsed = time.perf_counter() - chat_t0

                print(f"[Worker] Chat response: {text}")
                print(f"\n\n####chat finish time = {_now_str()} (took {chat_elapsed:.2f}s)")
                image, error = None, None
        except Exception as e:
            print(f"[Worker] Pipeline failed: {e}")
            image = None
            text = None
            error = str(e)

        with _results_lock:
            entry = _results[job_id]
            entry["image"] = image
            entry["text"] = text
            entry["error"] = error
            entry["done"].set()


def _start_worker():
    t = threading.Thread(target=_pipeline_worker, daemon=True, name="mlx-worker")
    t.start()
    return t


def submit_and_wait(prompt, timeout=3600):
    print("\n\n####submit to queue start time =", _now_str())

    """Enqueue a request, block until the worker classifies + processes it."""
    job_id = str(uuid.uuid4())
    done_event = threading.Event()
    with _results_lock:
        _results[job_id] = {"done": done_event, "image": None, "text": None, "error": None}

    _inbox.put({"id": job_id, "prompt": prompt})

    if not done_event.wait(timeout):  # timed out without a result
        return None, None, "Timed out waiting for processing."

    with _results_lock:
        entry = _results.pop(job_id, {"image": None, "text": None, "error": "No result"})
        print("\n\n####queue pop time =", _now_str())

    return entry["image"], entry["text"], entry["error"]


# ---- Request handling -------------------------------------------------
def return_image_response(image, filename="generated_image.png", keep=False):
    """Save the generated image to a temp file and return it to the client.

    The mflux image's .save() takes a path (str/PathLike), not a BytesIO.
    """
    import tempfile, os
    from flask import send_file

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    try:
        image.save(path=tmp_path)          # .save expects a path, not BytesIO
        return send_file(tmp_path, mimetype="image/png", as_attachment=True, download_name=filename)
    finally:
        if not keep:
            os.unlink(tmp_path)            # cleaned up after the response is sent


def get_request_data():
    """Parse the JSON body, requiring a non-empty 'prompt'. Returns dict or None."""
    try:
        data = request.get_json(silent=False)  # already a parsed dict
        if not data:
            print("No JSON data provided")
            return None
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            print("No valid 'prompt' provided")
            return None
        
        print(f"\n###Received request data: {data}", f"####received data time ={_now_str()}")
        return data
    except Exception as e:
        print(f"Error occurred while fetching request data: {e}")
        return None


@app.route("/", methods=["GET", "POST"])
def hello():
    data = get_request_data()
    if data is None:
        return jsonify({"status": "error", "message": "Missing valid 'prompt' in JSON body."}), 400

    prompt = data["prompt"]
    image, text, error = submit_and_wait(prompt)

    if error is not None:
        return jsonify({"status": "error", "message": error}), 500

    if image is not None:
        return return_image_response(image)
    return jsonify({"status": "ok", "answer": text})


@app.route("/<path:path>", methods=["GET", "POST"])
def catch_all(path):
    data = get_request_data()
    if data is None:
        return jsonify({"status": "error", "message": "Missing valid 'prompt' in JSON body."}), 400

    prompt = data["prompt"]
    image, text, error = submit_and_wait(prompt)

    if error is not None:
        return jsonify({"status": "error", "message": error}), 500

    if image is not None:
        return return_image_response(image)
    return jsonify({"status": "ok", "answer": text, "path": path})

# Your unified worker function that now accepts the device choice
def _start_worker(device_type):
    if device_type == "cuda":
        print("🚀 Starting atomic worker on NVIDIA CUDA backend...")
        # Add your CUDA model loading logic here
    elif device_type == "mlx":
        print("🍏 Starting atomic worker on Apple Silicon MLX backend...")
        # Add your MLX model loading logic here
    else:
        print(f"❌ Unknown device type: {device_type}")
        sys.exit(1)


# ---- Startup -----------------------------------------------------------
if __name__ == "__main__":
    # 1. Setup the command-line argument parser
    parser = argparse.ArgumentParser(description="Run the text/image generation server.")
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "mlx"], 
        default="mlx",  # Defaults to MLX if you don't pass anything
        help="Specify the hardware backend: 'cuda' for NVIDIA or 'mlx' for Apple Silicon."
    )
    
    # 2. Parse the arguments
    args = parser.parse_args()
    
    # Start the single atomic MLX worker (loads models, classifies, generates).
    _start_worker()

    # threaded=True lets many clients connect; the real work is serialized
    # on the one worker thread above.
    # host="0.0.0.0" accepts connections from other PCs on the network.
    app.run(host="0.0.0.0", port=8000, threaded=True, use_reloader=False)
