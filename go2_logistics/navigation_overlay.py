"""Inset shows sensor observations and planned path, not authored obstacles."""
def draw_navigation(frame,navigator,root):
    import cv2
    import numpy as np
    height,width=frame.shape[:2]
    x0,y0=width-230,height-200
    frame[y0:height,x0:width]=(22,29,35)
    def pixel(x,y):
        return (int(x0+10+(x+8)/16*210),int(y0+178-(y+2)/13*148))
    for x,y,layer in navigator.grid.observed:
        px,py=pixel(x*navigator.grid.resolution,y*navigator.grid.resolution)
        if x0<=px<width and y0<=py<height:
            cv2.circle(frame,(px,py),1,(245,175,65),-1)
    if navigator.events:
        event=navigator.events[-1]
        points=np.array([pixel(*p) for p in [event['start']]+event['path']],dtype=np.int32)
        cv2.polylines(frame,[points],False,(55,210,220),2)
    cv2.circle(frame,pixel(*root[:2]),4,(85,245,100),-1)
    cv2.putText(frame,'SENSED MAP / A*',(x0+8,y0+17),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1)
    cv2.putText(frame,f'Scans {len(navigator.scans)} | plans {navigator.plans}',(x0+8,height-7),cv2.FONT_HERSHEY_SIMPLEX,.36,(240,240,240),1)
