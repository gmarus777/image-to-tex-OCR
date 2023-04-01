
import requests
from PIL import Image
import cv2
import streamlit
import numpy as np
from pathlib import Path
import sys
from torchvision import transforms
import torch
import json



from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Data.image_transforms import Image_Transforms
from Tokenizer.Tokenizer import token_to_strings
from Models.Printed_Tex_Transformer import ResNetTransformer
from Lightning_Models.Printed_Tex_Lit_Model import LitResNetTransformer






if __name__ == '__main__':
    streamlit.set_page_config(page_title='LaTeX OCR Model')
    streamlit.title('LaTeX OCR')
    streamlit.markdown('Convert Math Formula Images to LaTeX Code.\n\nBased on the `image-to-tex-OCR` project. For more details  [![github](https://img.shields.io/badge/image--to--Tex--OCR-visit-a?style=social&logo=github)](https://github.com/gmarus777/image-to-tex-OCR)')

    uploaded_image = streamlit.file_uploader(
        'Upload Image',
        type=['png', 'jpg'],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        streamlit.image(image)
        image = np.asarray(image)
        image_tensor = Image_Transforms.test_transform_with_padding(image=np.array(image))['image'][:1]

    if streamlit.button('Convert'):
        if uploaded_image is not None and image_tensor is not None:
            files = {"file": uploaded_image.getvalue()}
            with streamlit.spinner('Converting Image to LaTeX'):
                response = requests.post('http://127.0.0.1:8000/predict/', files={'file': uploaded_image.getvalue()})

            latex_code = response.json()["data"]["pred"]
            streamlit.code(latex_code, language='latex')
            streamlit.markdown(f'$\\displaystyle {latex_code}$')

        else:
            streamlit.error('Please upload an image.')




    # Method without server
    #if streamlit.button('Convert'):
        #if upload is not None and image_tensor is not None:
            #with streamlit.spinner('Computing'):
               # scripted = torch.jit.load("Models_Parameters_Log/scripted_model1.pt")
               # my_prediction = scripted(image_tensor.unsqueeze(0))
               # latex_code = token_to_strings(tokens = my_prediction)

               # streamlit.title('Result')
                #streamlit.markdown('LaTeX Code:')
                #streamlit.code(latex_code, language='latex')
                #streamlit.markdown('Rendered LaTeX:')
                #streamlit.markdown(f'$\\displaystyle {latex_code}$')

        #else:
            #streamlit.error('Please upload an image.')

        # show images
        #transform = transforms.ToPILImage()
        #streamlit.image(image)
        #streamlit.image(transform(image_tensor))



    #else:
        #streamlit.text('\n')





   #


