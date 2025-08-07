from imx500_zoo.datasets.default_posenet import DefaultPosenet

class ArrowPosenet(DefaultPosenet):
    DOWNLOAD_DATASET = r"https://github.com/SonySemiconductorSolutions/aitrios-rpi-dataset-sample/raw/main/arrow_posenet.zip"
    ZIP_SUBFOLDER = "arrow"
    ZIP_FILENAME = "arrow.zip"
