import gradio as gr
from openai import OpenAI
from openai import AuthenticationError, RateLimitError, APIConnectionError, OpenAIError, Timeout
from  prompt_generation import get_aug_prompt


def get_response(prompt, api_key):
    try:        
        client = OpenAI(api_key = api_key)
        # Add the user's question to the messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]

        # Get the response from the model
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Get the assistant's reply
        reply = response.choices[0].message.content.strip()

        return reply
    
    except AuthenticationError:
        return "Invalid API key provided. Please check your API key and try again."
    except RateLimitError:
        return "Rate limit exceeded. Please wait and try again later."
    except APIConnectionError:
        return "Network error. Please check your connection and try again."
    except Timeout:
        return "Request timed out. Please try again."
    except OpenAIError as e:
        return f"An error occurred: {e}"



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale = 1):
            query_input = gr.Textbox(lines=2, placeholder="Enter your query here...", label="User Query")
            model_dropdown = gr.Dropdown(choices=['Dual Encoder GAT'], label="Choose a Model")
            api_key_input = gr.Textbox(placeholder="Enter your OpenAI API key here...", label="API Key", type="password")
            submit_btn = gr.Button("Submit")
            original_response = gr.Textbox(label="Original Response")

        with gr.Column(scale = 1):
            prompt_output = gr.Textbox(label="Augmented Prompt")
            augmented_output = gr.Textbox(label="Augmented Response")

        def get_model_response(query, model, api_key):
            reply_original = get_response(query, api_key)
            augmented_prompt = get_aug_prompt.main(query)
            reply_aug = get_response(augmented_prompt, api_key)
            # return augmented_prompt, reply_original, reply_aug
            return {
                    original_response: reply_original, 
                    prompt_output: augmented_prompt, 
                    augmented_output: reply_aug}

        submit_btn.click(
            get_model_response,
            [query_input, model_dropdown, api_key_input],
            [original_response, prompt_output, augmented_output],
        )

demo.launch(server_name="129.108.18.37", server_port=7860, share=True)

# prompt = "Can you suggest common weaknesses and vulnerabilities related to the Colonial Pipeline Attack? In May of 2021, a hacker group known as DarkSide gained access to Colonial Pipeline’s network through a compromised VPN password. This was possible, in part, because the system did not have multifactor authentication protocols in place. This made entry into the VPN easier since multiple steps were not required to verify the user’s identity. Even though the compromised password was a “complex password,” malicious actors acquired it as part of a separate data breach."