import argparse
from MacTmp import CPU_Temp
import os
import pyautogui as pag
import time

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

########## Params ##########
CODE_COUNTRY = 'BGD'#'ARG'#'AUS'#'AND'
COORDS_NEXT = (180, 130)
SIZE_IMG = (32, 32)  #like CFAR10   #(204, 102)#(144, 144)
SIZE_DB = 6000  #like CFAR10    #1000
STARTING_IDX = args.start
dt = 2#5#.1
############################

########## Aux fs ##########
def click_next(): 
  pag.click(*COORDS_NEXT)
  time.sleep(3)

# def create_dir():
#   if 'res' not in os.listdir():
#     os.mkdir('res')

def edit_img(img):
  w, h = img.size
  img = img.crop((0, 350, w, h-150))
  img = img.resize(SIZE_IMG)
  return img

def save_img(img, fn): 
  img.save(f'res/imgs/{CODE_COUNTRY}/' + fn + '.png')
  time.sleep(dt)

def take_screenshot(): 
  img = pag.screenshot()
  time.sleep(dt)
  return img

def check_T_range(T):
  if T > 85:#75: 
    pag.move(0, 100)
    raise Exception('T too high!!')


############################

def main():
  time.sleep(2)
  # create_dir()

  for i in range(STARTING_IDX, SIZE_DB):
    T = float(CPU_Temp())
    print(T)
    check_T_range(T)
    while T > 75:#60:
      pag.move(100, 0)
      time.sleep(60)
      T = float(CPU_Temp())
      print(f'Stopped. New T: {T}')
      check_T_range(T)

    fn = f'{CODE_COUNTRY}_' + f'{i}'.zfill(6)
    img = take_screenshot()
    img = edit_img(img)
    save_img(img, fn)
    click_next()

if __name__ == '__main__':
  t0 = time.time()
  main()
  t1 = time.time()
  print(f'Execution time: {t1 - t0 :.1f}s')




































