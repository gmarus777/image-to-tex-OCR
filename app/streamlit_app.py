import requests
from PIL import Image
import streamlit


streamlit_app.set_page_config(page_title='LaTeX-OCR')
streamlit_app.title('LaTeX OCR')
streamlit_app.markdown('Convert images of equations to corresponding LaTeX code.\n\nThis is based on the `pix2tex` module. Check it out [![github](https://img.shields.io/badge/LaTeX--OCR-visit-a?style=social&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR)')

uploaded_file = streamlit_app.file_uploader(
    'Upload an image an equation',
    type=['png', 'jpg'],
)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        streamlit_app.image(image)
    else:
        streamlit_app.text('\n')

    if streamlit_app.button('Convert'):
        if uploaded_file is not None and image is not None:
            with streamlit_app.spinner('Computing'):
                response = requests.post('http://127.0.0.1:8502/predict/', files={'file': uploaded_file.getvalue()})
            if response.ok:
                latex_code = response.json()
                streamlit_app.code(latex_code, language='latex')
                streamlit_app.markdown(f'$\\displaystyle {latex_code}$')
            else:
                streamlit_app.error(response.text)
        else:
            streamlit_app.error('Please upload an image.')