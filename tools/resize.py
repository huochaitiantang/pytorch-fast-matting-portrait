from PIL import Image
import sys

img = Image.open(sys.argv[1])
#img.show()

alignw = 600
alignh = 800

w,h = img.size
print(w,h)

upratio = max(alignw/float(w), alignh/float(h))

if upratio > 1.0:
    w = int(w * upratio)
    h = int(h * upratio)
    print(w, h)
    img = img.resize((w,h),Image.ANTIALIAS)
    #img.show()

assert(alignw <= w and alignh <= h)

if alignw != w or alingh != h:
    cw = w / 2
    ch = h / 2
    x1 = cw - alignw / 2
    y1 = ch - alignh / 2
    rect = (x1, y1, x1 + alignw, y1 + alignh)
    img = img.crop(rect)
    print(img.size)
    #img.show()
    img.save(sys.argv[1]+'_new.jpg')


