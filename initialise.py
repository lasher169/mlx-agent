import re
from datetime import datetime

# Optional / heavy MLX imports are done lazily inside the functions that need
# them (load_models, generate_image, send) so that `import test` always works,
# even before mlx/mflux are installed or while models are still downloading.

def _now_str():
    """Wall-clock timestamp with millisecond precision, for log lines."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


model_1, tokenizer_1, model_2, tokenizer_2, flux, image = None, None, None, None, None, None

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


def keyboard_input_worker(q):
    while True:
        try:
            # input() will raise EOFError if Ctrl+D is pressed
            user_text = input("Enter something (or press Ctrl+D / type 'exit'): ")
            q.put(user_text)
            
            if user_text.lower() == 'exit':
                break
                
        except EOFError:
            # Handle Ctrl+D gracefully
            print("\n[Input Thread]: Ctrl+D detected. Sending exit signal...")
            q.put("exit")
            break
        
def load_models_cuda():
    global model_1, tokenizer_1, flux

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from diffusers import FluxPipeline
    
    # --- MODEL 1: Text Model (Llama-3.1-8B 8-bit equivalent) ---
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # 8-bit quantization configuration for CUDA
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer_1 = AutoTokenizer.from_pretrained(model_id)
    model_1 = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="cuda"  # Automatically maps to NVIDIA GPU
    )
    print("Loaded Model 1 on CUDA")

    # --- FLUX: Image Model (Flux.1 Schnell) ---
    # To mimic quantize=4 or 'schnell' 4-bit, we load the pipeline in bfloat16.
    # Note: For strict 4-bit weights on Flux via CUDA, look up `bghira/Flux-schnell-nf4`
    flux_id = "black-forest-labs/FLUX.1-schnell"
    
    flux = FluxPipeline.from_pretrained(
        flux_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Loaded Flux on CUDA")    

def load_models_mlx():
    global model_1, tokenizer_1, model_2, tokenizer_2, flux, image
    from mlx_lm import load
    from mflux.models.z_image import ZImageTurbo

    # Load Model 1 (e.g., a small text model)
    # model_1, tokenizer_1 = load("mlx-community/Qwen3.8-27B-4bit")
    #model_1, tokenizer_1 = load("mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit")
    # model_1, tokenizer_1 = load("mlx-community/gemma-4-26B-A4B-it-qat-OptiQ-4bit", lazy=False)
    model_1, tokenizer_1 = load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit", lazy=False)
    
    print("load model1")

    # Load Model 2 (e.g., a coding or secondary model)
    # model_2, tokenizer_2 = load_img("mlx-community/FLUX.2-klein-9B")

    print("load model2")
    
    flux = ZImageTurbo(quantize=4) 
    # flux = Flux1.from_name(model_name="schnell", quantize=4)    

def generate_image(data):
    global flux, image
    
    print(f"\n[Main]: Generating image with prompt: {data}")
    
    image = flux.generate_image(
        prompt=data,
        num_inference_steps=40, # 40 steps is recommended by Alibaba for Qwen-Image
        seed=42
    )
    
    # image = flux.generate_image(
    #     seed=42,
    #     prompt=data,
    #     num_inference_steps=4,
    #     height=1024,
    #     width=1024
    # )
    return image

def send(data, model_type):
    from mlx_lm import generate

    messages = [{"role": "user", "content": f"{data}"}]
    # 4. Turn the conversation dictionary into the model's expected tokens
    formatted_prompt = tokenizer_1.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    
    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("####Current Time1 =", current_time)
    
    # print(f"\n[Main]: Sending data to {model_type}: {data}")
    # Use Model 1
    response_1 = generate(model_1, 
        tokenizer_1, 
        prompt=formatted_prompt, 
        verbose=False,)
    # print(f"\n[Model 1 Response]: {response_1}")
    
    # The thinking model emits '<|channel>thought ... <channel|>answer';
    # return only the final answer, not the reasoning channel.
    return _strip_thinking(response_1)


