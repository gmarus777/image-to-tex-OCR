
import requests
from PIL import Image
import streamlit


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


if __name__ == '__main__':
    streamlit.set_page_config(page_title='LaTeX OCR Model',
                              layout="centered",
                              menu_items={
                                  'Get Help': 'https://github.com/gmarus777/image-to-tex-OCR',
                                  'Report a bug': "https://github.com/gmarus777/image-to-tex-OCR",
                                  'About': "# LaTeX *OCR* Model"
                              }
                              )

    streamlit.title('LaTeX OCR')
    streamlit.markdown('Convert Math Formula Images to LaTeX Code.\n\nBased on the `image-to-tex-OCR` project. For more details  [![github](https://img.shields.io/badge/image--to--Tex--OCR-visit-a?style=social&logo=github)](https://github.com/gmarus777/image-to-tex-OCR)')

    uploaded_image = streamlit.file_uploader(
        'Upload Image',
        type=['png', 'jpg'],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        streamlit.image(image)
        #image = np.asarray(image)
        #image_tensor = Image_Transforms.test_transform_with_padding(image=np.array(image))['image'][:1]

    if streamlit.button('Convert'):
        if uploaded_image is not None: #and image_tensor is not None:
            files = {"file": uploaded_image.getvalue()}
            with streamlit.spinner('Converting Image to LaTeX'):

                #Docker image
                response = requests.post('http://0.0.0.0:8000/predict/', files={'file': uploaded_image.getvalue()})

            latex_code = response.json()["data"]["pred"]
            streamlit.code(latex_code, language='latex')
            streamlit.markdown(f'$\\displaystyle {latex_code}$')

        else:
            streamlit.error('Please upload an image.')







