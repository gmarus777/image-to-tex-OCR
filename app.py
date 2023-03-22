import requests
from PIL import Image
import cv2
import streamlit
import numpy as np
from  Data.image_transforms import Image_Transforms
from Models.Printed_Tex_Transformer import ResNetTransformer
from torchvision import transforms
import torch
import json




def findPositions(image):
    positions = np.nonzero(image)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    image = cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 0)
    return image


def load_dic(filename):
    with open(filename) as f:
        dic = json.loads(f.read())
        dic_new = dict((k, int(v)) for k, v in dic.items())
    return dic_new

def token_to_strings(tokens):
    skipTokens = {'<S>', '<E>', '<P>'}
    mapping = load_dic('Data/Data_Bank/230k.json')
    inverse_mapping =invert_vocabulary(mapping)
    s=''
    if tokens.shape[0] ==1:
        tokens = tokens[0]
    for number in tokens:
        letter = inverse_mapping[number.item()]
        if letter not in skipTokens:
            s= s +" " + str(letter)
    return s

def invert_vocabulary(vocabulary):
    inverse_vocabulary = {}
    for letter, idx in vocabulary.items():
        inverse_vocabulary[idx] = letter
    return inverse_vocabulary

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
        image = cv2.bitwise_not(image)
        image = findPositions(image)
        h, w, c = image.shape
        aspect = h / w

        # Thresholding
        if w > 400:
            ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Downscaling big images
        if w > 1200:
            new_w = 1000
            new_h = int(new_w * aspect)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


        image_tensor = Image_Transforms.test_transform_with_padding_medium(image=np.array(image))['image'][:1]

    # Initiate model and predict image
    if streamlit.button('Convert'):
        if upload is not None and image_tensor is not None:
            with streamlit.spinner('Computing'):
                dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model = ResNetTransformer(max_label_length=130, num_classes= 579).to(dev)
                model.load_state_dict(torch.load(("Models_Parameters_Log/Printed1_2D600_350.pth"), map_location=torch.device('cpu')))
                model.eval()
                my_prediction = model.predict(image_tensor.unsqueeze(0).to(dev))
                latex_code = token_to_strings(my_prediction)

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


