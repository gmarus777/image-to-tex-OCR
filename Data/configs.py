from pathlib import Path



# TODO: need to document and rename all directories names



class PrintedLatexDataConfig:
    DATA_DIRNAME = Path(__file__).resolve().parents[0]   # gives the "root/Data" directory
    DATA_BANK_DIRNAME = DATA_DIRNAME / "Data_Bank"
    PNG_IMAGES_NAMES_FILE = DATA_BANK_DIRNAME / 'corresponding_png_images.txt'
    PNG_FINAL_FORMULAS = DATA_BANK_DIRNAME / "final_png_formulas.txt"
    GENERATED_PNG_DIR_NAME = DATA_BANK_DIRNAME / "generated_png_images"



    # Handwritten

    HANDWRITTEN_TRAIN = DATA_BANK_DIRNAME /'train.lst'
    HANDWRITTEN_VAL =  DATA_BANK_DIRNAME /'val.lst'
    HANDWRITTEN_FORMULAS = DATA_BANK_DIRNAME / 'formulas.norm.lst'
    HANDWRITTEN_IMAGES_FOLDER = DATA_BANK_DIRNAME/'images'








