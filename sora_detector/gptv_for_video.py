# %% [markdown]
# # Processing and narrating a video with GPT's visual capabilities and the TTS API
# 
# This notebook demonstrates how to use GPT's visual capabilities with a video. GPT-4 doesn't take videos as input directly, but we can use vision and the new 128K context window to describe the static frames of a whole video at once. We'll walk through two examples:
# 
# 1. Using GPT-4 to get a description of a video
# 2. Generating a voiceover for a video with GPT-4 and the TTS API
# 


import argparse
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
import PIL
import yaml
base_url = "https://gateway.ai.cloudflare.com/v1/c1d85cabb875f2197b5568185ceb8ba4/ai-gateway/openai"
api_key = "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"
client = OpenAI(api_key=api_key, base_url=base_url)


def run(args):
    directory = args.img_path

# List to store the base64 encoded images
    base64_images = []
    key_images = []
    key_images_cluster = {}
    key_index = "0"
    files = os.listdir(directory)
    image_files = sorted([f for f in files if f.endswith('.jpg') or f.endswith('.png')], key=lambda x: int(x.split('_')[0]))
# Iterate over every file in the directory
    for filename in image_files:
    # We only want to process images
        
        # Open the image file
        with open(os.path.join(directory, filename), 'rb') as image_file:
            # Read the image file
            image_data = image_file.read()
            
            # Convert the image data to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            # Add the base64 image to the list
            base64_images.append(base64_image)
            
            tokens = filename.split(".")[0].split("_")
            
            if len(tokens) == 2:
                 key_images.append(base64_image)
            if tokens[1] not in key_images_cluster:
                 key_images_cluster[tokens[1]] = []
                 print(filename)
            key_images_cluster[tokens[1]].append(base64_image)
            # if tokens[1] == key_index:
            #      key_images_cluster[key_index].append(base64_image)
            # else:
            #      key_index = tokens[1]
            #      key_images_cluster[key_index] = []

    print("Extract Images Success\n####################################################\n")
    # print(len(key_images),len(key_images_cluster["key1"]),len(key_images_cluster["key2"]),len(key_images_cluster["key3"]),len(key_images_cluster["key4"]),len(key_images_cluster["key5"]))
    # key_images = base64_images[1::3]
    with open("./prompt.yaml","r",encoding='utf-8') as file:
            prompt = yaml.load(file, yaml.FullLoader)
    # key_images_cluster = []
    # for i in range(0, len(base64_images), 3):
    #     key_images_cluster.append(base64_images[i:i+3])
    origin_prompt = "A burning candle was blown out by a gust of wind."
    generated_prompt = "In the image, there is a toy doll shown close-up, lying on a creamy beige fabric surface. The doll is depicted wearing a purple top and blue pants. The background and doll details are slightly blurred, suggesting a depth of field commonly used in photography and film to focus attention on the subject. This image has a soft, serene quality which may evoke themes of childhood, innocence, or tranquility. While this frame alone doesn't reveal any action or significant object changes, it appears to be part of a sequence intended to convey a gentle, possibly nostalgic narrative involving the toy doll. Without additional frames or explicit changes in key objects from dynamic knowledge graphs, it's challenging to provide a broader storyline or progression."
    # PROMPT_MESSAGES = [
    #     {
    #         "role": "user",
    #         "content": [
    #             *map(lambda x: {"image": x, "resize": 768}, key_images),
    #             prompt['inconsistency_comparison'].format(generated_prompt=generated_prompt, origin_prompt=origin_prompt),
    #         ],
    #     },
    # ]
    # params = {
    #     "model": "gpt-4-turbo",
    #     "messages": PROMPT_MESSAGES,
    #     "max_tokens": 4096,
    # }
    # print("Prompt: " + prompt['inconsistency_comparison'].format(generated_prompt=generated_prompt, origin_prompt=origin_prompt) + "\n####################################################\n")
    # for key_image in key_images:
    #     PROMPT_MESSAGES = [
    #         {
    #         "role": "user",
    #         "content": [
    #             prompt['static_KG'],
    #             *map(lambda x: {"image": x, "resize": 768}, [key_image]),
    #         ],
    #         },
    #     ]
    #     params = {
    #         "model": "gpt-4-turbo",
    #         "messages": PROMPT_MESSAGES,
    #         "max_tokens": 4096,
    #     }
    #     print("Prompt: " + prompt['static_KG'] + "\n####################################################\n")
    #     result = client.chat.completions.create(**params)
    #     print(result.choices[0].message.content)
    #     print("\n####################################################\n")

    # relation = """{"objects": ["basketball", "basketball_hoop", "basketball_net", "light_pole", "slide", "fence"],"relations": [["basketball", "entering", "basketball_net"],["basketball_net", "attached to", "basketball_hoop"],["basketball_hoop", "mounted on", "backboard"],["slide", "located at", "playground"],["fence", "surrounds", "basketball_court"]]}\n"""
    relation = """"""
    # prompt_ = prompt['dynamic_KG'].format(relation=relation)
    # PROMPT_MESSAGES = [
        
    #     {
    #         "role": "user",
    #         "content": [
    #             prompt_,
    #             *map(lambda x: {"image": x, "resize": 768}, key_images_cluster["key1"]),
    #         ],
    #     },
    # ]
    # params = {
    #     "model": "gpt-4-turbo",
    #     "messages": PROMPT_MESSAGES,
    #     "max_tokens": 4096,
    # }
    # print("Prompt: " + prompt_ + "\n####################################################\n")
    # prompt_ = prompt['static_hall_detection'].format(relation=relation, origin_prompt=origin_prompt)
    # PROMPT_MESSAGES = [
        
    #     {
    #         "role": "user",
    #         "content": [
    #             prompt_,
    #             *map(lambda x: {"image": x, "resize": 768}, [key_images[0]]),
    #         ],
    #     },
    # ]
    # params = {
    #     "model": "gpt-4-turbo",
    #     "messages": PROMPT_MESSAGES,
    #     "max_tokens": 4096,
    # }
    # print("Prompt: " + prompt_ + "\n####################################################\n")

    # change = """{{"changes": {"basketball": "moves downward through the net", "basketball_hoop": "unchanged", "basketball_net": "unchanged", "light_pole": "unchanged", "slide": "unchanged", "fence": "unchanged"}}}"""
    change = """"""
    changes = """"""
    # PROMPT_MESSAGES = [
    #     {
    #         "role": "user",
    #         "content": [
    #             *map(lambda x: {"image": x, "resize": 768}, key_images),
    #             prompt['video_summary'].format(changes=changes),
    #         ],
    #     },
    # ]
    # params = {
    #     "model": "gpt-4-turbo",
    #     "messages": PROMPT_MESSAGES,
    #     "max_tokens": 4096,
    # }
    # print("Prompt: " + prompt['video_summary'].format(changes=changes) + "\n####################################################\n")
    prompt_ = prompt['dynamic_hall_detection'].format(relation = relation, change=change, origin_prompt=origin_prompt)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                *map(lambda x: {"image": x, "resize": 768}, key_images_cluster["key1"]),
                prompt_
            ],
        },
    ]
    params = {
        "model": "gpt-4-turbo",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1024,
    }
    print("Prompt: " + prompt_ + "\n####################################################\n")
    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)



if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str,  default="../sora_detector/sora_video_keyframe_group/21", help="root folder for input")
    args = parser.parse_args()
    run(args)