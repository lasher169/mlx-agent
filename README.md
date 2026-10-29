# mlx-server

This is an attempt to build a Python app which connects directly to the LLM models. 

The chat model is configured through `CHAT_MODEL` in `.env`. The MLX default is `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`; the CUDA loader uses `meta-llama/Meta-Llama-3.1-8B-Instruct` when `CHAT_MODEL` is unset. Use a model identifier compatible with the selected backend.

## Why I wrote this
I wrote this because I can load multiple models at once. I can also run image + chat models or any other models that float my boat. 

I then expose this as a web server so people can post a prompt and the app will then either chat or build an image. This should theoretically be able to identify whether it's a chat or image request and then use the appropriate LLM. Because it's a web app, it should also be accessible via `curl` or any other app that can hit the endpoint.

I am also under the perception it is faster. :)

## MLX.Support

At the moment it supports mlx but i think you can also use CUDA to do something similiar. I will see what I can do to make it work with CUDA. 

## How to run

do a pip install -r requirements-cuda.txt or the mlx equivalent. this should install the required libs for you.

there is .env.example file change the model to the one you want to use. eg  CHAT_MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-8bit

Run the server with `python server.py --device mlx` on Apple Silicon or `python server.py --device cuda` on a CUDA machine. Install the matching requirements file first.

To change the chat model, edit `CHAT_MODEL` in `.env`. `.env.example` shows the expected format, and `.env` is excluded from Git.

