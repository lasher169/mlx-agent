# mlx-agent

This is an attempt to build a Python app which connects directly to the LLM models. 

At the moment, you need to change the models you want to use in the code, but I will extend it so it will read from a configuration file. The `test.py` file is responsible for loading the models.

## Why I wrote this
I wrote this because I can load multiple models at once. I can also run image + chat models or any other models that float my boat. 

I then expose this as a web server so people can post a prompt and the app will then either chat or build an image. This should theoretically be able to identify whether it's a chat or image request and then use the appropriate LLM. Because it's a web app, it should also be accessible via `curl` or any other app that can hit the endpoint.

I am also under the perception it is faster. :)

## MLX.Support

At the moment it supports mlx but i think you can also use CUDA to do something similiar. I will see what I can do to make it work with CUDA. 