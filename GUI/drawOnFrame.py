from PIL import Image, ImageDraw, ImageFont
import cv2, numpy, csv
import ffmpy

# get the video
capture = cv2.VideoCapture("C:\\Anju\\Final Year Project\\Stanford drone dataset\\videos\\coupa\\video0\\video.mov")
files = "bookstore0train\\bookstore0train_"
files= "deathCircle0train\\deathCircle0train_"
files="C:\\Anju\\Final Year Project\\coupa\\frame\\"

height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
video = cv2.VideoWriter('coupa.avi', -1, 1, (width, height))
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# video = cv2.VideoWriter('coupa.mp4',fourcc,15.0,(width, height))
i = 0
background = Image.open('marker_dot.png').convert('RGBA')
bg_w, bg_h = background.size
pointsActual = []
pointsPredicted =[]
while (capture.get(cv2.CAP_PROP_POS_FRAMES) < frame_count - 1):
    ret, frame1 = capture.read()
    # base = Image.fromarray(numpy.uint8(frame1)).convert('RGBA')
    s = open(files + str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))) + ".csv")
    reader = csv.reader(s)
    readerIterator = reader.__iter__()
    base = frame1
    # print(int('123.3456'))
    if(i<40):
        i+=1
        continue
    for s in readerIterator:
        # print(s)
        pointsActual.append((int(round(float(s[3].strip()))), int(round(float(s[4].strip())))))
        pointsPredicted.append((int(round(float(s[1].strip()))), int(round(float(s[2].strip())))))
    for j in pointsActual:
        base = cv2.circle(frame1, j, 1, (255, 0, 0), 2)
    for j in pointsPredicted:
        base1 = cv2.circle(base, j, 1, (0, 255, 0), 2)
        # base.paste(background, j)
    # points.append(points[-1]+bg_w)
    # video.write(numpy.asarray(base))
    video.write(base1)

    # print(i)
    i += 1
    if i == 150:
        break
cv2.destroyAllWindows()
video.release()
