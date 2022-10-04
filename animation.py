import pygame.gfxdraw
import pygame
from enum import Enum
import math
import imageio
import os

class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (200, 0, 0)
    YELLOW = (170, 180, 0)
    LIGHTGRAY = (211,211,211)
    GRAY = (180, 180, 180)
    LIGHTBLUE = (150, 150, 220)


class BallAnimation():

    def __init__(self, speed = 10, y_min_real = 0.0, y_max_real=1.0, setpoint=0.5):
        """_summary_

        Args:
            speed (int): Animation speed. Defaults to 10.            
            y_min_real (float): Minimal real (from model) position.
            y_max_real (float): Maximal real (from model) position.
        """
        # window and pendulum characteristics
        self.width = 360
        self.height = 480
        self.speed = speed
        self.center = (self.width/2, self.height/2)
        self.ball_position = list(self.center)  # inicijalno pozicija lopte je na sredini, mora se konvertovati u listu jer tuplovi ne mogu da se azuriraju
        self.ball_position_max = (self.width/2, self.height - 40)
        self.ball_position_min = (self.width/2, 60)
        self.y_min_real = y_min_real
        self.y_max_real = y_max_real
        self.setpoint = setpoint
        
        pygame.init()
        self.display = pygame.display.set_mode(size = (self.width, self.height))
        self.clock = pygame.time.Clock()   
        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        self.my_font = pygame.font.SysFont('Comic Sans MS', 10)
        self.filenamelist = []   # filenames for images needed to save animation to gif
        self.frame = 0

        
    def plot_ball_position(self, new_pos):
        """Plots ball position on windiow

        Args:
            new_pos (float): Ball position
        """
        self.ball_position[1] = int((self.ball_position_max[1]-self.ball_position_min[1])/(self.y_max_real - self.y_min_real)*new_pos + self.ball_position_min[1])

        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                quit()
 
        self.display.fill(Color.BLACK.value) 
        self.plot_setpoint(self.setpoint)
        # steel plate drawing
        plate_coord = [(10, self.height - 20), (10, self.height - 10), 
                        (self.width - 10, self.height - 10), (self.width - 10, self.height - 20)]
        pygame.gfxdraw.filled_polygon(self.display, plate_coord, Color.LIGHTGRAY.value)
        # coil drawing
        coil_shell_coord = [(10, 50), (10, 10), (self.width - 10, 10), (self.width - 10, 50),
                            (self.width - 20, 50), (self.width - 20, 20), (20, 20), (20, 50)]
        pygame.gfxdraw.filled_polygon(self.display, coil_shell_coord, Color.LIGHTGRAY.value)
        coil_coord = [(30, 20), (self.width - 30, 20), (self.width - 30, 40), (30, 40)]
        pygame.gfxdraw.filled_polygon(self.display, coil_coord, Color.GRAY.value)
        pygame.gfxdraw.polygon(self.display, coil_coord, Color.WHITE.value)

        # firstly, filled circle representig ball is drawn, and than its edges are drawn with antialiaser
        pygame.gfxdraw.filled_circle(self.display, int(self.ball_position[0]), int(self.ball_position[1]), 20, Color.YELLOW.value)
        pygame.gfxdraw.aacircle(self.display, int(self.ball_position[0]), int(self.ball_position[1]), 20, Color.YELLOW.value)      
        # pygame.display.update() # updates changes 
        # self.clock.tick(self.speed)


    def plot_setpoint(self, setpoint):
        """Draws line that indicates setpoint

        Args:
            setpoint (float): Setpoint value
        """
        ycoord = int((self.ball_position_max[1]-self.ball_position_min[1])/(self.y_max_real - self.y_min_real)*setpoint + self.ball_position_min[1])
        pygame.gfxdraw.line(self.display, 10, ycoord, self.width-10, ycoord, Color.RED.value)
 
    def plot_current(self, i, imax):
        """Draws rectangle that indicates current intensity.

        Args:
            i (float): Input current
            imax (float): Maximum current
        """
        y = int((self.height*4/5-self.center[1])/(imax - 0)*i + self.center[1])  # crtamo od pola do 4/5
        pygame.gfxdraw.filled_polygon(self.display, [(10, self.center[1]), (10, y), (20, y), (20, self.center[1])], Color.LIGHTBLUE.value)
        text = self.my_font.render(f'i={round(i, 2)}', False, Color.LIGHTBLUE.value)
        self.display.blit(text, (10, self.height*4/5 + 10))


    def plot_voltage(self, V, Vmax):
        """Draws rectangle that indicates voltage intensity.

        Args:
            V (float): Input voltage
            Vmax (float): Maximum voltage
        """
        y = int((self.height*4/5-self.center[1])/(Vmax - 0)*V + self.center[1])  # crtamo od pola do 4/5
        pygame.gfxdraw.filled_polygon(self.display, [(self.width - 10, self.center[1]), (self.width - 10, y), (self.width - 20, y), (self.width - 20, self.center[1])], Color.LIGHTBLUE.value)
        text = self.my_font.render(f'V={round(V)}', False, Color.LIGHTBLUE.value)
        self.display.blit(text, (self.width - 30, self.height*4/5 + 10))

    def update(self, export_to_gif = False):
        """Updates drawing surface.

        Args:
            export_to_gif (bool): Saves frames to images if flag i True for further exporting to GIF. Defaults to False.
        """
        pygame.display.update()
        self.clock.tick(self.speed)
        
        if export_to_gif:
            self.filenamelist.append(f"pic{self.frame}.png")
            pygame.image.save(self.display, self.filenamelist[self.frame])
            self.frame += 1
        

    def export_to_gif(self):
        """Creates GIF file with animation. Before calling this method, update() methode should be called
        with export_to_gif parameter set to True, so that frames are exported to images.
        """
        images = []
        for filename in self.filenamelist:
            images.append(imageio.imread(filename))
        imageio.mimsave("ball_animation.gif", images, duration = 0.02)
        
        #Remove the PNG files (if they were meant to be temporary):
        for filename in self.filenamelist:
            os.remove(filename)
  