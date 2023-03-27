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

MAX_RATIO =15

def findPositions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = 255*(gray < 50).astype(np.uint8)  # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))  # Perform noise filtering
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    # Crop the image - note we do this on the original image
    cropped_image = image[y-10:y+h+10, x-10:x+w+10]

    return cropped_image


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
        # image = findPositions(image)

        h, w, c = image.shape
        ratio = int(w / h)
        if ratio == 0:
            ratio = 1
        if ratio > MAX_RATIO:
            ratio = MAX_RATIO

        # Thresholding
        #if w > 400:
            #ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        new_h = 128
        new_w = int(new_h * ratio)
        if h > 128:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        image_tensor = Image_Transforms.test_transform_with_padding(image=np.array(image))['image'][:1]

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


