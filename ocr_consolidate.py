import os
import json
import base64
import requests
import openai
from PIL import Image
from dotenv import load_dotenv,find_dotenv
from utils import arguements,init_logger

def raise_res(
    img_dir:str,
    thres:int = 100
)->None:
    resized_path = os.path.join(img_dir,"resized")
    if not os.path.exists(resized_path):
        os.mkdir(resized_path)
        
    for img_f in os.listdir(img_dir):
        if not img_f.endswith(".png"):
            continue
        img_path = os.path.join(img_dir,img_f)
        logger.info(img_path)
        img = Image.open(img_path,'r')
        width, height = img.size
        logger.info(f"SIZE:{width} x {height}")
        if height < thres:
            img.close()
            os.remove(img_path)
            continue
    
        new_width = img.width * 2  # Double the width
        new_height = img.height * 2  # Double the height
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        img.close()
        width, height = img_resized.size
        logger.info(f"RESIZE:{width} x {height}")
        img_resized.save(os.path.join(resized_path,img_f))
   
def read_response()->dict:
    file = 'table_images/1372807/2024-03-31/output/soi_table_0.json'
    with open(file, 'r') as f:
        data = json.load(f)
        print(data['choices'][0]['message']['content'])
    
def to_gpt(
    qtr_dir:str,
    api_key:str
)->tuple:
    if not os.path.exists(os.path.join(qtr_dir,"output")):
        os.mkdir(os.path.join(qtr_dir,"output"))
        
    for img in os.listdir(qtr_dir):
        if not img.endswith(".png"):
            continue
        img_path = os.path.join(qtr_dir,img)
        logger.info(f"GPT getting json {img_path}")
        img_64 = encode_image(img_path)
        gpt_response = get_json(img_64,api_key)
        logger.info(gpt_response)
        # table = json.loads(gpt_response['choices'][0]['message']['content'])
        save_path = os.path.join(qtr_dir.split("resized")[0],"output",f"{img.split('.png')[0]}.json")
        logger.info(f"SAVING to  {save_path}")
        with open(save_path,'w') as f:
            json.dump(gpt_response,f,indent=4)
    

def encode_image(image_path:str)->base64:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')   


def get_json(
    img_bytes:base64,
    api_key:str,
)->dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are an intelligent system api that reads the texts from an image and outputs the texts as key values pairs in JSON format. Given an image, output the texts in JSON format."
                },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_bytes}"
                        }
                }
            ]
        }
    ],
        "max_tokens": 10000
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()
  
def main()->None:
    env_path = find_dotenv('.env')
    load_dotenv(env_path)
    openai.api_key = api_key = os.getenv("API_KEY")
    
    table_images = os.path.join('table_images',args.cik)
    for qtr in os.listdir(table_images):
        qtr = '2024-03-31'
        qtr_dir = os.path.join(table_images,qtr)
        logger.info(qtr_dir)
        raise_res(qtr_dir)
        to_gpt(os.path.join(qtr_dir,"resized"),api_key)
        break
    

if __name__ == '__main__':
    args = arguements()
    logger = init_logger(args.cik)
    # main()
    read_response()
    