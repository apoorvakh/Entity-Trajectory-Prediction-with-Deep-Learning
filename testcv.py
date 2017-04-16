import cv2

video_stream = cv2.VideoCapture()
video_stream.open('C:\\video.mov')
if not video_stream.isOpened():
    raise RuntimeError( "Error when reading image file")
  
total_frame_count = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
fps= video_stream.get(cv2.CAP_PROP_FPS)
height = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
print(total_frame_count, fps, height, width)
