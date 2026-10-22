import threading
import queue
import time
from datetime import datetime

from server import _classify, _strip_thinking, _now_str
# Optional / heavy MLX imports are done lazily inside the functions that need
# them (load_models, generate_image, send) so that `import test` always works,
# even before mlx/mflux are installed or while models are still downloading.


model_1, tokenizer_1, model_2, tokenizer_2, flux, image = None, None, None, None, None, None


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
        
    

def load_models():
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
        num_inference_steps=20, # 40 steps is recommended by Alibaba for Qwen-Image
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
    

def main():
    print("Hello, World mofo!")
    load_models()
    
    # Create a thread-safe queue to pass text between threads
    input_queue = queue.Queue()
    
    input_thread = threading.Thread(target=keyboard_input_worker, args=(input_queue,), daemon=True)
    input_thread.start()
    
    print("\nSystem ready. Type commands below:")
    
        # 3. Main execution loop
    while True:
        # Check if the keyboard thread sent any data
        if not input_queue.empty():
            data = input_queue.get()
            
            if data.lower() == 'exit':
                print("[Main]: Exit signal received. Shutting down system...")
                break
            
            print("\n\n####start time =", _now_str())

            print(f"[Main processed command]: {data}")
            # prompt = send(data, model_type="model1")
            kind = _classify(data, send)
            
            print(f"[Classfication] → {kind})")
            
            # image = generate_image(data)
            # image.save(path="my_generated_image.png")

            print("####Finish time =", _now_str())
            print("\nSystem ready. Type commands below:")
            
        # Simulate your main loop doing background model tasks or inferences
        # print(".", end="", flush=True)
        time.sleep(0.1)
        


if __name__ == "__main__":
    main()
