import requests
from PIL import Image
import streamlit

if __name__ == '__main__':
    streamlit.set_page_config(page_title='LaTeX OCR')
    streamlit.title('LaTeX OCR')
    streamlit.markdown('Convert Math Formula Images to LaTeX Code.\n\nBased on the `image-to-tex-OCR` project. For more details  [![github](https://img.shields.io/badge/image--to--Tex--OCR-visit-a?style=social&logo=github)](https://github.com/gmarus777/image-to-tex-OCR)')

    uploaded_file = streamlit.file_uploader(
        'Upload image equation',
        type=['png', 'jpg'],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        streamlit.image(image)
    else:
        streamlit.text('\n')

    if streamlit.button('Convert'):
        if uploaded_file is not None and image is not None:
            with streamlit.spinner('Computing'):
                response = requests.post('http://127.0.0.1:8502/predict/', files={'file': uploaded_file.getvalue()})
            if response.ok:
                latex_code = response.json()
                streamlit.code(latex_code, language='latex')
                streamlit.markdown(f'$\\displaystyle {latex_code}$')
            else:
                streamlit.error(response.text)
        else:
            streamlit.error('Please upload an image.')