import sys
from os.path import abspath
from cv2 import imwrite

from util import load_ir_in_dat

if __name__ == '__main__':
    files = './ginseng-ir/irimage_20220728_0903.dat'
    file_path = abspath(files)
    img = load_ir_in_dat(file_path)
    diff = img.max() - img.min()
    img = (img - img.min()) / diff
    img *= 255
    img = img.astype('uint8')
    imwrite('test.jpg', img)
