
import requests
from PIL import Image
import cv2
import streamlit
import numpy as np
from Data.image_transforms import Image_Transforms
from Models.Printed_Tex_Transformer import ResNetTransformer
from Lightning_Models.Printed_Tex_Lit_Model import LitResNetTransformer
from torchvision import transforms
import torch
import json
from Stream_app.Tokenizer import token_to_strings





if __name__ == '__main__':
    streamlit.set_page_config(page_title='LaTeX OCR Model')
    streamlit.title('LaTeX OCR')
    streamlit.markdown('Convert Math Formula Images to LaTeX Code.\n\nBased on the `image-to-tex-OCR` project. For more details  [![github](https://img.shields.io/badge/image--to--Tex--OCR-visit-a?style=social&logo=github)](https://github.com/gmarus777/image-to-tex-OCR)')

    upload = streamlit.file_uploader(
        'Upload Image',
        type=['png', 'jpg'],
    )

    if upload is not None:
        image = Image.open(upload).convert('RGB')
        streamlit.image(image)
        image = np.asarray(image)
        image_tensor = Image_Transforms.test_transform_with_padding(image=np.array(image))['image'][:1]

    # Initiate model and predict image
    if streamlit.button('Convert'):
        if upload is not None and image_tensor is not None:
            with streamlit.spinner('Computing'):
                scripted = torch.jit.load("Models_Parameters_Log/scripted_model1.pt")
                my_prediction = scripted(image_tensor.unsqueeze(0))
                latex_code = token_to_strings(tokens = my_prediction)

                streamlit.title('Result')
                streamlit.markdown('LaTeX Code:')
                streamlit.code(latex_code, language='latex')
                streamlit.markdown('Rendered LaTeX:')
                streamlit.markdown(f'$\\displaystyle {latex_code}$')

        else:
            streamlit.error('Please upload an image.')

        # show images
        #transform = transforms.ToPILImage()
        #streamlit.image(image)
        #streamlit.image(transform(image_tensor))



    else:
        streamlit.text('\n')





   # if streamlit.button('Convert'):
      #  if uploaded_file is not None and image is not None:
      #      with streamlit.spinner('Computing'):
      #          response = requests.post('http://127.0.0.1:8502/predict/', files={'file': uploaded_file.getvalue()})
      #      if response.ok:
       #         latex_code = response.json()
       #         streamlit.code(latex_code, language='latex')
        #        streamlit.markdown(f'$\\displaystyle {latex_code}$')
        #    else:
        #        streamlit.error(response.text)
       # else:
       #     streamlit.error('Please upload an image.')


