from cmath import nan
from urllib.response import addinfo
import numpy as np
import pygame
import sys 
import math

BLACK = (0, 0, 0)
BLUE=(0, 0, 200)
RED=(200, 0, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 1201
WINDOW_WIDTH = 1201
GRID_WIDTH=5
GRID_HEIGHT=5
MARGIN=1
UP=0
DOWN=1
LEFT=2
RIGHT=3
VELOCITY=20
LENGTH_RAY=100
OPTIMAL_RANGE=90
ALPHA=10
nRay=6
OCCUPY=1.4
L_PRIOR=-3
L_FREE=0

class robot:
    #ray is a np.array to save the end point of the ray
    def __init__(self, x, y, ray, theta=0) -> None:
        self.x=x
        self.y=y
        self.theta=theta
        self.ray=ray
        self.width=GRID_WIDTH
        self.height=GRID_HEIGHT
        self.v=0
        self.angle=0

    def updateDubin(self, angle, dt):
        self.x=self.x+np.cos(angle)*dt*VELOCITY
        self.y=self.y+np.sin(angle)*dt*VELOCITY
        self.theta=angle
        self.ray[0]=np.mod(self.ray[0]+angle, 2*np.pi)
        self.ray[1]=np.mod(self.ray[1]+angle, 2*np.pi)


    def update(self, dir):
        if dir==DOWN:
            self.updateDubin(np.pi/2, 0.1)
        elif dir==UP:
            self.updateDubin(1.5*np.pi, 0.1)
        elif dir==LEFT:
            self.updateDubin(np.pi, 0.1)
        else:
            self.updateDubin(0, 0.1)
    def getEndPoints(self):
        #ray=np.zeros((len(self.ray), 2))
        #for i, r in enumerate(self.ray):
        r=self.ray[0]
        #end1=[np.cos(r)*LENGTH_RAY+self.x, np.cos(r)*LENGTH_RAY+self.y]
        r2=self.ray[1]
        #end2=[np.cos(r2)*LENGTH_RAY+self.x, np.cos(r2)*LENGTH_RAY+self.y]
        return np.array([[np.cos(r)*LENGTH_RAY+self.x, np.cos(r)*LENGTH_RAY+self.y], [np.cos(r2)*LENGTH_RAY+self.x, np.cos(r2)*LENGTH_RAY+self.y]])

            

    def getLoc(self, dir):
        self.update(dir)
        return self.x, self.y






global SCREEN, CLOCK, color


def draw(width, height, color, SCREEN):
    SCREEN.fill(WHITE)
    for r in range(height):
        for c in range(width):
            g=color[r, c]
            if g>1 or math.isnan(g):
                g=0.9
            g=(1-g)*200 #grayscale:0 to 1
            #print("The current g: ", g)
            pygame.draw.rect(SCREEN, (g, g, g), [(MARGIN + GRID_WIDTH) * c + MARGIN, (MARGIN + GRID_HEIGHT) * r + MARGIN, GRID_WIDTH, GRID_HEIGHT])

#def GenGrid(blockSize=10):
#    for x in range(0, WINDOW_WIDTH, blockSize):
#        for y in range(0, WINDOW_HEIGHT, blockSize):
#            rectangle = pygame.Rect(x, y, blockSize, blockSize)
#            pygame.draw.rect(SCREEN, BLACK, rectangle, 1)

#Return the grids hit by the line
#This needs optimized, but o.k. for now
def HitY(start, end,  k, r, c):
    x=(MARGIN+GRID_WIDTH)*c+MARGIN
    y=start[1]+k*(x-start[0])
    if (y<=(MARGIN+GRID_WIDTH)*(r+1)+MARGIN and y>=(MARGIN+GRID_WIDTH)*r+MARGIN) and (x<=end[0]) and (x>=start[0]):
        return True
    else:
        return False

def HitX(start, end, k, r, c):
    #If it is not a block, just return
    y=(MARGIN+GRID_WIDTH)*r+MARGIN
    x=start[0]+(y-start[1])/k
    if (x<=(MARGIN+GRID_WIDTH)*(c+1)+MARGIN and y>=(MARGIN+GRID_WIDTH)*c+MARGIN) and (x<=end[0]) and (x>=start[0]):
        return True
    else:
        return False

def SegHit(color, start, end):
    ReColor=color
    k=(end[1]-start[1])/(end[0]-start[0])
    for r in range(200):
        for c in range(200):
            if color[r][c]==1:
                continue
            hit1=HitX(start, end, k, r, c)
            hit2=HitX(start, end, k, r, c+1)
            hit3=HitY(start, end, k, r, c)
            hit4=HitY(start, end, k, r+1, c)
            if hit1 or hit2 or hit3 or hit4:
                ReColor[r][c]=0
    return ReColor

def getAngle(start, end):
    return np.mod(np.arctan2(end[0]-start[0], end[1]-start[1]), 2*np.pi)

def angleAttack(angle, start, end):
    phi=getAngle(start, end)
    left=phi>=angle[0]
    right=phi<=angle[1]
    checkLen=np.linalg.norm(end-start, 2)<=LENGTH_RAY
    return left and right and checkLen

def angleCheck(rob, r, c):
    start=np.array([rob.x, rob.y])
    c1=np.array([(MARGIN+GRID_WIDTH)*r+MARGIN, (MARGIN+GRID_WIDTH)*c+MARGIN])
    c2=np.array([(MARGIN+GRID_WIDTH)*(r+1)+MARGIN, (MARGIN+GRID_WIDTH)*c+MARGIN])
    c3=np.array([(MARGIN+GRID_WIDTH)*r+MARGIN, (MARGIN+GRID_WIDTH)*(c+1)+MARGIN])
    c4=np.array([(MARGIN+GRID_WIDTH)*(r+1)+MARGIN, (MARGIN+GRID_WIDTH)*(c+1)+MARGIN])
    ck1=angleAttack(rob.ray+rob.theta, start, c1)
    ck2=angleAttack(rob.ray+rob.theta, start, c2)
    ck3=angleAttack(rob.ray+rob.theta, start, c3)
    ck4=angleAttack(rob.ray+rob.theta, start, c4)
    return ck1 or ck2 or ck3 or ck4

def CastToLog(color):
    return np.log(color)-np.log(1-color)

def CastBack(color):
    return np.exp(color)/(np.exp(color)+1)

def SegHitOGM(curr_color, rob):
    color=CastToLog(curr_color) #initialze to 0, as log(0.5/0.5)=0
    for r in range(200):
        for c in range(200):
            #If the grid is unblocking or already blocked
            #if color[r, c]==1 or color[r, c]==0:
            #    continue 
            #if the grid is within the perception field
            #OGM
            end=np.array([(MARGIN+GRID_WIDTH)*(r+0.5)+MARGIN, (MARGIN+GRID_WIDTH)*(c+0.5)+MARGIN])
            if angleCheck(rob, r, c):
                #If the range of the grid IS within the optimal range, then compute the inverse sensor model
                if np.abs(np.linalg.norm(np.array([rob.x-end[0], rob.y-end[1]]),2)-OPTIMAL_RANGE)<=ALPHA:
                    color[r, c]=color[r, c]+OCCUPY-L_PRIOR
                #If the range of the grid is GREATER than the optimal range then...take as PRIOR
                elif np.linalg.norm(np.array([rob.x-end[0], rob.y-end[1]]),2)-OPTIMAL_RANGE>ALPHA:
                    color[r, c]+=L_PRIOR
                elif np.linalg.norm(np.array([rob.x-end[0], rob.y-end[1]]),2)-OPTIMAL_RANGE<-ALPHA:
                    #color[r, c]+=L_FREE-L_PRIOR 
                    color[r, c]+=L_PRIOR-L_FREE
            #if the grid is outside the perception field
            else:
                continue
    #recast to probability:
    return CastBack(color)
            

if __name__ == "__main__":
    color=np.ones((200, 200))*0.5
#    color=np.random.rand(40,40)
#    for r in range(len(color)):
#        for c in range(len(color[0])):
#            if color[r, c]<0.5:
#                color[r, c]=0.5
#            else:
#                color[r, c]=1
    #ray=np.random.rand(nRay)*2*np.pi
    phi1=np.random.rand()
    phi2=phi1+4
    ray=np.mod(np.array([phi1, phi2]), np.pi)
    #print(ray)
    rob=robot(300, 300, ray)
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    IsClick=False
    while True: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_LEFT:
                    px, py=rob.getLoc(LEFT)
                if event.key==pygame.K_RIGHT:
                    px, py=rob.getLoc(RIGHT)
                if event.key==pygame.K_UP:
                    px, py=rob.getLoc(UP)
                if event.key==pygame.K_DOWN:
                    px, py=rob.getLoc(DOWN)
                IsClick=True
        draw(200, 200, color, SCREEN)
        pygame.draw.rect(SCREEN, (255, 0, 0), [rob.x, rob.y, GRID_WIDTH, GRID_HEIGHT]) 
        #pos=np.array([rob.x, rob.y])
        if IsClick:
            color=SegHitOGM(color, rob) #recolor the block hit by the ray
            #print(color)
        ends=rob.getEndPoints()
        #print(ends)
        pygame.draw.line(SCREEN, BLUE, (rob.x, rob.y), (ends[0, 0], ends[0, 1]), 1)
        pygame.draw.line(SCREEN, RED, (rob.x, rob.y), (ends[1, 0], ends[1, 1]))
        IsClick=False
        pygame.display.update()
        pygame.time.delay(100)

