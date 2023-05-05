"""
Purpose:
    Story Slam
"""
import os
# 3rd party imports
import streamlit as st
import requests
import numpy as np
import ai21

endpoint_name = "j2-grande-instruct"

st.set_page_config(layout="wide")

# ENV Vars
text_api = os.environ["TEXT_API"]
text_key = os.environ["TEXT_KEY"]


image_api = os.environ["IMAGE_API"]
image_key = os.environ["IMAGE_KEY"]


# Initialization
if "story" not in st.session_state:
    st.session_state["story"] = []


def gen_text_api(data):
    # API endpoint
    api_endpoint = text_api

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": text_key,
    }

    resp = requests.post(api_endpoint, json=data, headers=headers)
    # print(resp.json())
    return resp.json()["text"]


def gen_image_api(text):
    api_endpoint = (
        image_api
    )

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": image_key,
    }

    data = {"text": text}

    resp = requests.post(api_endpoint, json=data, headers=headers)
    # print(resp.json())

    return np.array(resp.json()["generated_image"])


model_info = {
    "AI21 Jurassic-2 Grande Instruct": "Jurassic-2 Grande Instruct is a mid-sized language model carefully designed to strike the perfect balance between exceptional quality and affordability. You can use it to compose human-like text and solve complex language tasks such as question answering, text classification and many others. Learn more here: https://docs.ai21.com/docs/jurassic-2-models",
    "Flan T5-XL": "The Flan T5-XL model is a state-of-the-art, large-scale Text-To-Text Transfer Transformer that excels in various NLP tasks by employing a unified text-to-text framework with over 700 million paramaters. Learn more here: https://huggingface.co/google/flan-t5-xl",
    "Stable Diffusion 2.1": "Stable Diffusion 2.1 is a latent diffusion model that can generate images based on text inputs. Learn more here: https://huggingface.co/stabilityai/stable-diffusion-2-1",
}


def gen_text(col):
    """
    Purpose:
        Controls the text gen process
    Args:
        N/A
    Returns:
        N/A
    """

    col.subheader("Text Generation Model")
    selected_model = col.selectbox(
        "Select Text Model", ["AI21 Jurassic-2 Grande Instruct", "Flan T5-XL"]
    )

    col.write(model_info[selected_model])

    if selected_model == "Flan T5-XL":
        # Model Options
        MAX_LENGTH = col.number_input(
            "Max Length",
            min_value=150,
            max_value=4000,
            value=500,
            help="Max Length of text returned",
        )

        NUM_RETURN_SEQUENCES = col.number_input(
            "Number of Return Sequences",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of generated text sequences to return",
        )

        TOP_K = col.number_input(
            "Top K",
            min_value=0,
            max_value=100,
            value=0,
            help="The number of highest probability vocabulary tokens to keep for top-k-filtering",
        )

        TOP_P = col.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="The cumulative probability of parameter highest probability tokens to keep for nucleus sampling",
        )

        DO_SAMPLE = col.checkbox(
            "Do Sample",
            value=True,
            help="Whether to use sampling or greedy approach for text generation",
        )

        prompt = col.text_area("Prompt:", height=300)

        model_data = {
            "max_length": MAX_LENGTH,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "do_sample": DO_SAMPLE,
            "prompt": prompt,
        }
        if col.button("Generate"):
            with st.spinner("In progress..."):
                generated_text = gen_text_api(model_data)

                col.write(generated_text)
                # Add to the story so far
                story_json = {
                    "prompt": prompt,
                    "text": generated_text,
                    "model": selected_model,
                }
                st.session_state["story"].append(story_json)
    if selected_model == "AI21 Jurassic-2 Grande Instruct":
        MAX_LENGTH = col.number_input(
            "Max Length",
            min_value=5,
            max_value=2000,
            value=500,
            help="Max Length of text returned",
        )

        TEMPERATURE = col.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="You can increase creativity by tweaking the temperature. With temperature 0, the model will always choose the most probable completion, so it will always be the same. Increasing the temperature will provide varying completions, where the completion may be different with every generation. ",
        )

        NUM_RETURN_SEQUENCES = col.number_input(
            "Number of Return Sequences",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of generated text sequences to return",
        )

        prompt = col.text_area("Prompt:", height=300)

        if col.button("Generate"):
            with st.spinner("In progress..."):
                response = ai21.Completion.execute(
                    sm_endpoint=endpoint_name,
                    prompt=prompt,
                    maxTokens=MAX_LENGTH,
                    temperature=TEMPERATURE,
                    numResults=NUM_RETURN_SEQUENCES,
                )

                gen_text = ""
                for comp in response["completions"]:
                    gen_text += comp["data"]["text"].strip()

                    gen_text += "\n\n End of Response \n\n"

                col.write(gen_text)
                # Add to the story so far
                story_json = {
                    "prompt": prompt,
                    "text": gen_text,
                    "model": selected_model,
                }
                st.session_state["story"].append(story_json)


def gen_image(col):
    """
    Purpose:
        Controls the image gen process
    Args:
        N/A
    Returns:
        N/A
    """
    col.subheader("Image Generation Model")
    selected_img_model = col.selectbox("Select Image Model", ["Stable Diffusion 2.1"])

    col.write(model_info[selected_img_model])

    image_prompt = col.text_area("Image Prompt:", height=300)

    if col.button("Generate"):
        with st.spinner("In progress..."):
            generated_image = gen_image_api(image_prompt)

            col.image(generated_image)

            # Add to the story so far
            story_json = {
                "prompt": image_prompt,
                "image": generated_image,
                "model": selected_img_model,
            }
            st.session_state["story"].append(story_json)


# def the_story_so_far():
def render_story(col):
    for index, item in enumerate(st.session_state.story):
        if "text" in item:
            col.markdown(f"**Prompt:** {item['prompt']}")
            col.write(item["text"])
            # TODO will need to play test
            if col.button("Remove Text from Story", key=f"text_reomve_index_{index}"):
                del st.session_state.story[index]
                st.experimental_rerun()

        elif "image" in item:
            col.markdown(f"**Prompt:** {item['prompt']}")
            col.image(item["image"])
            if col.button("Remove Image from Story", key=f"image_reomve_index_{index}"):
                del st.session_state.story[index]
                st.experimental_rerun()


def app() -> None:
    """
    Purpose:
        Controls the app flow
    Args:
        N/A
    Returns:
        N/A
    """

    gen_mode = st.selectbox("Select Generation Mode", ["Text", "Image"])

    col1, col2 = st.columns(2)

    if gen_mode == "Text":
        gen_text(col1)
    elif gen_mode == "Image":
        gen_image(col1)
    else:
        st.error("Invalid mode")

    col2.subheader("The story so far...")

    # Show the story so far
    render_story(col2)


def main() -> None:
    """
    Purpose:
        Controls the flow of the streamlit app
    Args:
        N/A
    Returns:
        N/A
    """

    # Start the streamlit app

    st.markdown(
        "<h1 style='text-align: center;'>Generative AI Story Slam</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='text-align: center;'>Unleash Your Imagination with AI-Powered Storytelling</h3>",
        unsafe_allow_html=True,
    )

    st.write("Come up with a creative story based on a theme leveraging foundation models hosted on AWS. You goal will be to generate a story of up to 3000 characters, and up to 5 images. You are free to edit text afterwards, just make sure to include the original prompt and output. You will then share your story with the audience. Your story will be judged based on prompt engineering, story delivery, and audience sentiment. Now let's build!")

    app()


if __name__ == "__main__":
    main()
