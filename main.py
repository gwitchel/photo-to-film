# maybe look into googles teachable machine 
from PIL import Image
from skimage.io import imread
import numpy as np
from Palette import Pallette
from Converter import Converter
from PaletteTransfer import PaletteTransfer
import pickle
# c = Converter(source_img="my_film/Scan 19.jpeg",palette_size=100)
# c.convert("images_to_convert/07_26_23_Mickeys_059.jpg")
# c.show_target()

palette = Pallette(["euphoria/4.jpeg"],10)
filehandler = open("euphoria"+"_palette","wb")
pickle.dump(palette,filehandler)
filehandler.close()
# file = open("lomography_1"+"_palette",'rb')
# palette = pickle.load(file)
# file.close()

# palette.save_list_of_bin_colors()
transfer = PaletteTransfer(palette)
transfer.convert("images_to_convert/E4DB57F0-374A-4D1A-82FC-82079525CB63.jpg")

# palette2 = Pallette("my_film/Scan 19.jpeg",palette_size=50)
