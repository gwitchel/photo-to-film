import numpy as np
import math 
from PIL import Image
import pickle
from itertools import product
import math 
class Pallette: 
    size = None
    colors_bin = None
    palette_map = None
    black_threshold = None
    occurrence_threshold = None # minimum amount of times a pixel needs to be counted to occur 
    def __init__(self,src_images,size=20,occurrence_threshold=10,black_threshold=120) -> None:
        print("Initializing a palette from" , src_images, "With size", size)
        self.size = size 
        self.black_threshold = black_threshold
        self.occurrence_threshold = occurrence_threshold
        self.colors_bin = [[[0 for i in range (256)] for j in range(256)] for k in range(256)]
        self.palette_map = [[[-1 for i in range (256)] for j in range(256)] for k in range(256)]
        
        for src_image in src_images:
            im1 = Image.open(src_image)
            im1_arr = np.asarray(im1)
            src_px = [[i for i in arr] for arr in im1_arr] 

            self.populate_colors_bin(src_px)
        self.generate_palette_map()
    
    def generate_palette_map(self):
        print("Beginning generation of palette map: ", self.size**3, " Cubes to match")
        ''' given a 3d vector map of colors, generate a hash map for each color'''
        percent_done = lambda p : (p/(255/self.size)**3)*100
        amount_completed = 0
        for r in range(0,255,self.size):
            for g in range(0,255,self.size):
                print(percent_done(amount_completed),"% Done")
                for b in range(0,255,self.size):
                    self.palette_map[r][g][b] = self.closest_palette_color(r,g,b)
                    amount_completed += 1 
   
    def closest_palette_color(self,r,g,b,bt=0):
        ''' breadth first search for closest pallette color to rgb'''
        
        directions = [(0,0,self.size),(0,self.size,0),(self.size,0,0),(0,0,-1*self.size),(0,-1*self.size,0),(-1*self.size,0,0)]
        queue = [(r,g,b)]
        history = [] 
        while len(queue) > 0 : 
            #1. pop the first value off the queue 
            (npr,npg,npb) = queue.pop()
            
            history.append((npr,npg,npb))
            if self.colors_bin[npr][npg][npb]: 
                # Avoid trending towards black 
                if max([npr,npg,npb]) <= self.black_threshold and max([r,g,b]) > self.black_threshold :
                    return self.closest_palette_color(r,g,b,self.black_threshold)
                else:
                    return [npr,npg,npg]
            
            #2. assuming found is false, add all directions to the queue 
            n_d = [(npr+d[0],npg+d[1],npb+d[2]) for d in directions]
            for d in n_d: 
                if max(d) < 255 and min(d) >= bt and d not in queue and d not in history:
                    queue.append(d)
        return [r,g,b]
    
    def color_dist(self,color1,color2):
        ''' gives the vector distance between two colors'''
        diff = (color1[0]-color2[0],color1[1]-color2[1],color1[2]-color2[2])
        return math.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)

    def populate_colors_bin(self, arr): 
        for row in arr: 
            for (c1,c2,c3) in row:
                
                c1 = c1 - c1 % self.size
                c2 = c2 - c2 % self.size
                c3 = c3 - c3 % self.size

                self.colors_bin[c1][c2][c3] += 1
        
        for row in arr: 
            for (c1,c2,c3) in row:
                if self.colors_bin[c1][c2][c3] < self.occurrence_threshold: 
                    self.colors_bin[c1][c2][c3] = 0 

    def save_list_of_palette_colors(self):
        ''' saves a list of palette colors for debugging purposes'''
        f = open("color_file.txt", "a")
        for r in range(self.size):
            for g in range(self.size):
                for b in range(self.size):
                    f.write(str(r)+","+str(g)+","+str(b)+"\n")
        f.close()
    
    def save_list_of_bin_colors(self):
        ''' saves a list of colors found in the originals image  for debugging purposes'''
        f = open("img_color_file.txt", "a")
        for r in range(self.size):
            for g in range(self.size):
                for b in range(self.size):
                    if(self.colors_bin[r][g][b]): f.write(str(r)+","+str(g)+","+str(b)+"\n")
        f.close()
    
    def combine_palettes(self,new_palette):
        ''' Takes in an additional palette and combines the two'''
        print("COMBINING PALETTES")
        if self.size != new_palette.size: 
            print("CANNOT COMBINE PALETTES THEY ARE DIFFERENT SIZES")
            pass 
        
        for r in range(self.size): 
            for g in range(self.size): 
                for b in range(self.size): 
                    d1 = self.color_dist(new_palette.palette_map[r][g][b] ,[r,g,b])
                    d2 =  self.color_dist(self.palette_map[r][g][b] ,[r,g,b])
                    
                    if d1 < d2: 
                        self.palette_map[r][g][b] = new_palette.palette_map[r][g][b]