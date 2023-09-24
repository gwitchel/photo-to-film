from Palette import Pallette
from PIL import Image
import numpy as np
import pickle

''' 
takes in a palette and transfers that palette onto a new image
'''
class PaletteTransfer: 
    palette = None
    mean_sqr_err = None

    def __init__(self,palette) -> None:
        self.palette = palette
        self.mean_sqr_err = None
    
    def convert(self,target_image): 
        print("Converting Image")
        im2 = Image.open(target_image)
        im2_arr = np.asarray(im2)
        self.target_px = [[i for i in arr] for arr in im2_arr] 

        out = [] 
        for n_row in range(len(self.target_px)):    
            mean_sq_err_l = []
            expand_color = lambda c: c - c % self.palette.size     
            row = []
            for color in self.target_px[n_row]: 

                c1 = expand_color(color[0])
                c2 = expand_color(color[1])
                c3 = expand_color(color[2])


                mapped = self.palette.palette_map[c1][c2][c3]
                if mapped == -1: 
                    mapped = [0,0,0]
                    print("NO MAP FOR", c1,c2,c3)

                new_c1 = mapped[0] + self.palette.size/2    
                new_c2 = mapped[1] + self.palette.size/2    
                new_c3 = mapped[2] + self.palette.size/2    

                mapped_color = np.array([new_c1,new_c2,new_c3],dtype="uint8")
                row.append(mapped_color)
                mean_sq_err_l.append(np.linalg.norm(mapped_color - np.array(color))**2)
                
            out.append(row)
        self.mean_sq_err = sum(mean_sq_err_l)/len(mean_sq_err_l)
        print("MSQ", self.mean_sq_err)
        img2 = Image.fromarray(np.asarray(out), 'RGB')
        img2.save('my.png')
        img2.show()

        return self.mean_sq_err, np.asarray(out)

