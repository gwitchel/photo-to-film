from Palette import Pallette
from PIL import Image
import numpy as np
import pickle
import PaletteTransfer
''' 
parent class for a palette transfer, handles multiples transfers etc. 
'''
class Converter: 
    img = None
    img_px = None 
    palettes=None
    conversions = {}
    def __init__(self,img) -> None: 
        self.img = img
        self.img_px = [[i for i in arr] for arr in np.asarray(Image.open(img))] 

    def show_source(self):
        img2 = Image.fromarray(np.asarray(self.src_px), 'RGB')
        img2.save('my.png')
        self.img.show()

    def map_palette_to_img(self,palette_file):
        '''loads palette file to palette, converts image, stores converted image as well as
        metrics on conversion '''
        file = open(palette_file,'rb')
        palette = pickle.load(file)
        file.close()

        transfer = PaletteTransfer(palette)
        mean_sq_err, converted_image = transfer.convert(self.img)

        self.conversions[self.img] = {
            "msq": mean_sq_err,
            "img": converted_image
        }
    
    # def show_best_conversion(self):




