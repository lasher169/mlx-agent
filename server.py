"""Flask HTTP gateway for the text and image models."""

import argparse
import os
import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_file

from initialise import _classify, generate_image, load_models_cuda, load_models_mlx, send

app = Flask(__name__)
_inbox = queue.Queue()
_results = {}
_results_lock = threading.Lock()


def _now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _pipeline_worker(device_type):
    if device_type == "cuda":
        load_models_cuda()
    else:
        load_models_mlx()

    print("\nModel loaded; worker thread ready to accept requests.\n")
    while True:
        job = _inbox.get()
        job_id = job["id"]
        prompt = job["prompt"]
        try:
            print(f"[Main processed command]: {prompt}")
            print("\n\n####classify start time =", _now_str())

            classify_t0 = time.perf_counter()
            kind = _classify(prompt, send)
            classify_elapsed = time.perf_counter() - classify_t0
            print(f"[Classification] -> {kind}")
            print(f"####Classify finish time = {_now_str()} (took {classify_elapsed:.2f}s)")

            if kind == "image":
                image_t0 = time.perf_counter()
                result_image = generate_image(prompt)
                image_elapsed = time.perf_counter() - image_t0
                result_image.save(path="my_generated_image.png")
                print(f"[Worker] Image done (took {image_elapsed:.2f}s)")
                text, error = None, None
            else:
                chat_t0 = time.perf_counter()
                text = send(prompt, model_type="model1")
                chat_elapsed = time.perf_counter() - chat_t0
                print(f"[Worker] Chat response: {text}")
                print(f"####Chat finish time = {_now_str()} (took {chat_elapsed:.2f}s)")
                result_image, error = None, None
        except Exception as exc:
            print(f"[Worker] Pipeline failed: {exc}")
            result_image = None
            text = None
            error = str(exc)

        with _results_lock:
            entry = _results[job_id]
            entry["image"] = result_image
            entry["text"] = text
            entry["error"] = error
            entry["done"].set()


def _start_worker(device_type):
    worker = threading.Thread(
        target=_pipeline_worker,
        args=(device_type,),
        daemon=True,
        name="model-worker",
    )
    worker.start()
    return worker


def submit_and_wait(prompt, timeout=3600):
    job_id = str(uuid.uuid4())
    done_event = threading.Event()
    with _results_lock:
        _results[job_id] = {
            "done": done_event,
            "image": None,
            "text": None,
            "error": None,
        }

    _inbox.put({"id": job_id, "prompt": prompt})
    if not done_event.wait(timeout):
        return None, None, "Timed out waiting for processing."

    with _results_lock:
        entry = _results.pop(job_id, {"image": None, "text": None, "error": "No result"})

    return entry["image"], entry["text"], entry["error"]


def return_image_response(result_image, filename="generated_image.png", keep=False):
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        result_image.save(path=tmp_path)
        return send_file(
            tmp_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=filename,
        )
    finally:
        if not keep:
            os.unlink(tmp_path)


def get_request_data():
    try:
        data = request.get_json(silent=False)
        if not data:
            return None
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        print(f"\n###Received request data: {data} ####received data time ={_now_str()}")
        return data
    except Exception as exc:
        print(f"Error occurred while fetching request data: {exc}")
        return None


@app.route("/", methods=["GET", "POST"])
def hello():
    data = get_request_data()
    if data is None:
        return jsonify({"status": "error", "message": "Missing valid 'prompt' in JSON body."}), 400

    result_image, text, error = submit_and_wait(data["prompt"])
    if error is not None:
        return jsonify({"status": "error", "message": error}), 500
    if result_image is not None:
        return return_image_response(result_image)
    return jsonify({"status": "ok", "answer": text})


@app.route("/<path:path>", methods=["GET", "POST"])
def catch_all(path):
    data = get_request_data()
    if data is None:
        return jsonify({"status": "error", "message": "Missing valid 'prompt' in JSON body."}), 400

    result_image, text, error = submit_and_wait(data["prompt"])
    if error is not None:
        return jsonify({"status": "error", "message": error}), 500
    if result_image is not None:
        return return_image_response(result_image)
    return jsonify({"status": "ok", "answer": text, "path": path})


def main():
    parser = argparse.ArgumentParser(description="Run the text/image generation server.")
    parser.add_argument(
        "--device",
        choices=["cuda", "mlx"],
        default="mlx",
        help="Hardware backend: cuda for NVIDIA or mlx for Apple Silicon.",
    )
    args = parser.parse_args()
    print(f"Starting model worker for {args.device} backend...")
    _start_worker(args.device)
    app.run(host="0.0.0.0", port=8000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()