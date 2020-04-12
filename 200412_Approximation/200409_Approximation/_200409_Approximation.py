import cv2
import numpy as np
import math
import queue
import time

pupils_L=[] #프레임에따른 pupil값 / left Queue
pupils_R=[]
#eyeSize_L=[np.array((0, 0))] # 눈 크기(w, h)
#eyeSize_R=[np.array((0, 0))]
currentEyeSize_L = np.array((0, 0)) # 눈 크기(w, h)
currentEyeSize_R = np.array((0, 0))
#W= [np.array([np.array([0,0]),np.array([0,0])]),np.array([np.array([0,0]),np.array([0,0])])]
W = [] # 홀수 행: x, 짝수 행: y / 현재(0, 0) = (L, R) => 나중에는 리스트나 특징 점 개수 지정
Wtilde = []
D = []
Vt = []

phi = 8.0 #tilt angle
x = 10.0 #Pe의 x좌표
f = 8.0 #focal length
H = 15 # 모니터 중심부터 카메라까지 높이차이

phi1 = np.zeros((3, 4))
phi1[0][0] = f * math.cos(phi)
phi1[0][2] = f * math.sin(phi)
phi1[0][3] = f * H * math.cos(phi)
phi1[1][1] = f
phi1[2][0] = -math.sin(phi)
phi1[2][2] = math.cos(phi)
phi1[2][3] = -H * math.sin(phi)

AbleDiff = 100
deleteDiff = 100
frameSave=30
PointNum = 2
WrowNum = 20
prevTime = 0

U_hat = np.empty((WrowNum, PointNum))
Vt_hat = np.empty((PointNum, WrowNum))
Rtemp = np.empty((4, PointNum))

cap = cv2.VideoCapture('video1_2.mp4')    #960,720

cx_recent = 0
cy_recent = 0
cx_r_recent = 0
cy_r_recent = 0
not_found = 0

#fps = cap.get(cv2.CAP_PROP_FPS) # 재생할 파일의 프레임 레이트 얻기
#fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 저장할 비디오 코덱
#filename = 'output1_2.avi' # 저장할 비디오 이름
#out = cv2.VideoWriter(filename, fourcc, fps, (int(w), int(h))) # 파일 stream 생성

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # 비디오 회전
        rows, cols = frame.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
        frame = cv2.warpAffine(frame, rotation_matrix, (cols, rows))
        # downsample
        #frame = cv2.pyrDown(cv2.pyrDown(frame))
       
        #detect face
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        path = "C:/Users/Admin/Downloads/opencv/sources/data/haarcascades/"
        eyes = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')

        detected = eyes.detectMultiScale(gray, 1.3, 5)

        faces = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
        detected_face = faces.detectMultiScale(gray, 1.3, 5)

        if len(detected) < 2 and not_found < 20: #지금 픽셀값 검사하기 그런식으로?..
            cv2.circle(frame,(int(cx_recent),int(cy_recent)),5,5,-1)
            cv2.circle(frame,(int(cx_r_recent),int(cy_r_recent)),5,5,-1)
            not_found = not_found + 1

        elif len(detected) == 2:
            if detected[0][0] > detected[1][0]: 
                (x_r,y_r,w_r,h_r)=detected[0]
                (x,y,w,h)=detected[1]
            else:
                (x_r,y_r,w_r,h_r)=detected[1]
                (x,y,w,h)=detected[0]

            #cv2.rectangle(frame,(x,y), (x + w, y + h),(0,0,255))
            #cv2.rectangle(frame,(x_r,y_r), (x_r + w_r, y_r + h_r),(0,0,255))
            cx,cy = int(x + w / 2), int(y + h / 2)
            if len(pupils_L) != 0:
                avg = sum(pupils_L)/len(pupils_L)
                diff = sum(abs(avg-(cx,cy)))

                if diff < AbleDiff :
                    pupils_L.insert(0,np.array([cx,cy])) # if문 안으로 넣었음 굳이 넣었다가 뺄 필요 없어서, 이 확정되지 않은 값으로 계산 포함시키지 않으려고

                    if len(pupils_L) == frameSave:
                        pupils_L.pop()
                    cv2.circle(frame,(int(cx),int(cy)),15,255,-1)
#                    cv2.line(frame, (cx,cy), (cx+int((2*sum(W)/len(W))[0][0]),cy+int((2*sum(W)/len(W))[0][1])), (100,200,100),1)
#                    cv2.line(frame, (cx,cy), (cx+int(W[2][0] - W[0][0]),cy+int(W[3][0] - W[1][0])), (100,200,100),1)
                    cx_recent = cx
                    cy_recent = cy
                    currentEyeSize_L = (w, h)
                elif diff <= deleteDiff :
                    pupils_L.insert(0,np.array([cx,cy])) # if문 안으로 넣었음 굳이 넣었다가 뺄 필요 없어서
                    if len(pupils_L) == frameSave:
                        pupils_L.pop()
            else:            
                pupils_L.insert(0,np.array([cx,cy]))
                cv2.circle(frame,(int(cx),int(cy)),15,255,-1)
#                cv2.line(frame, (cx,cy), (cx+int((2*sum(W)/len(W))[0][0]),cy+int((2*sum(W)/len(W))[0][1])), (100,200,100),1)
#                cv2.line(frame, (cx,cy), (cx+int(W[2][0] - W[0][0]),cy+int(W[3][0] - W[1][0])), (100,200,100),1)
                cx_recent = cx
                cy_recent = cy
                currentEyeSize_L = (w, h)

            cx_r,cy_r = int(x_r + w_r/2), int(y_r + h_r / 2)
            if len(pupils_R) != 0:
                avg_R = sum(pupils_R)/len(pupils_R)
                diff_R = sum(abs(avg_R-(cx_r,cy_r)))

                if diff_R < AbleDiff :
                    pupils_R.insert(0,np.array([cx_r,cy_r]))
                    if len(pupils_R) == frameSave:
                        pupils_R.pop()
                    cv2.circle(frame,(int(cx_r),int(cy_r)),15,255,-1)
#                    cv2.line(frame, (cx_r,cy_r), (cx_r+int((2*sum(W)/len(W))[1][0]),cy_r+int((2*sum(W)/len(W))[1][1])), (100,200,100),1)
#                    cv2.line(frame, (cx_r,cy_r), (cx_r+int(W[2][1] - W[0][1]),cy_r+int(W[3][1] - W[1][1])), (100,200,100),1)
                    cx_r_recent = cx_r
                    cy_r_recent = cy_r
                    currentEyeSize_R = (w_r, h_r)
                elif diff_R <= deleteDiff:
                    pupils_R.insert(0,np.array([cx_r,cy_r]))
                    if len(pupils_R) == frameSave:
                        pupils_R.pop()
            else:
                pupils_R.insert(0,np.array([cx_r,cy_r]))
                cv2.circle(frame,(int(cx_r),int(cy_r)),15,255,-1)
#                cv2.line(frame, (cx_r,cy_r), (cx_r+int((2*sum(W)/len(W))[0][0]),cy_r+int((2*sum(W)/len(W))[0][1])), (100,200,100),1)
#                cv2.line(frame, (cx_r,cy_r), (cx_r+int(W[2][1] - W[0][1]),cy_r+int(W[3][1] - W[1][1])), (100,200,100),1)
                cx_r_recent = cx_r
                cy_r_recent = cy_r
                currentEyeSize_R = (w_r, h_r)

        elif len(detected) > 2 and len(pupils_L) != 0 and len(pupils_R) != 0:
            avg = sum(pupils_L)/len(pupils_L)
            avg_R = sum(pupils_R)/len(pupils_R)
            minDiff = 10000
            minDiff_R = 10000
            minDct = detected[0]
            minDct_R = detected[0]

            for dct in detected:
                diff = sum(abs(avg - (dct[0] + dct[2]/2, dct[1] + dct[3]/2))) # dct[0] = x, dct[1] = y, dct[2] = w, dct[3] = h
                diff_R = sum(abs(avg_R - (dct[0] + dct[2]/2, dct[1] + dct[3]/2)))

                if minDiff > diff:
                    minDiff = diff
                    minDct = dct
                if minDiff_R > diff_R:
                    minDiff_R = diff_R
                    minDct_R = dct

            if minDiff <= deleteDiff:
                cx,cy = int(minDct[0] + minDct[2] / 2), int(minDct[1] + minDct[3] / 2)
                if minDiff < AbleDiff:
                    cv2.circle(frame,(int(cx),int(cy)),15,255,-1)
#                    cv2.line(frame, (cx,cy), (cx+int((2*sum(W)/len(W))[0][0]),cy+int((2*sum(W)/len(W))[0][1])), (100,200,100),1)
#                    cv2.line(frame, (cx,cy), (cx+int((2*sum(W)/len(W))[0][0]),cy+int((2*sum(W)/len(W))[0][1])), (100,200,100),1)
                    cx_recent = cx
                    cy_recent = cy
                    currentEyeSize_L = (w, h)
                pupils_L.insert(0, np.array([cx,cy]))
                if len(pupils_L) == frameSave:
                   pupils_L.pop()

            if minDiff_R <= deleteDiff:
                cx_r,cy_r = int(minDct_R[0] + minDct_R[2] / 2), int(minDct_R[1] + minDct_R[3] / 2)
                if minDiff_R < AbleDiff :
                    cv2.circle(frame,(int(cx_r),int(cy_r)),15,255,-1)
#                    cv2.line(frame, (cx_r,cy_r), (cx_r+int((2*sum(W)/len(W))[0][0]),cy_r+int((2*sum(W)/len(W))[0][1])), (100,200,100),1)
#                    cv2.line(frame, (cx_r,cy_r), (cx_r+int(W[2][1] - W[0][1]),cy_r+int(W[3][1] - W[1][1])), (100,200,100),1)
                    cx_r_recent = cx_r
                    cy_r_recent = cy_r
                    currentEyeSize_R = (w_r, h_r)
                pupils_R.insert(0, np.array([cx_r,cy_r]))
                if len(pupils_R) == frameSave:
                    pupils_R.pop()

            not_found = 0

  #      if len(pupils_L) > 1 and len(pupils_R) > 1:
 #           W.insert(0,np.array([np.array([0,0]),np.array([0,0])]))
 #           W[0][0]=np.array(pupils_L[0]-pupils_L[1]) # 행: 프레임, 열: 특징점, W[0][0] 안에 x, y 각각 존재
 #           W[0][1]=np.array(pupils_R[0]-pupils_R[1])
#            if len(W) > frameSave:
#                W.pop()

        W.insert(0,(pupils_L[0][0], pupils_R[0][0])) # L,R의 x좌표
        W.insert(0,(pupils_L[0][1], pupils_R[0][1])) # L,R의 y좌표
#        W[0][0]=np.array(pupils_L[0][0]) # 행: 프레임, 열: 특징점, W[0][0] 안에 x, y 각각 존재
#        W[0][1]=np.array(pupils_R[0][1])
        if len(W) > WrowNum:
           W.pop()
           W.pop()

        avg_row_x = sum(W[0]) / len(W[0])
        avg_row_y = sum(W[1]) / len(W[1])
        Wtilde.insert(0,(pupils_L[0][0] - avg_row_x, pupils_R[0][0] - avg_row_x)) # 행평균 빼서 넣어주기
        Wtilde.insert(0,(pupils_L[0][1] - avg_row_y, pupils_R[0][1] - avg_row_y)) # 행평균 빼서 넣어주기
        if len(Wtilde) > WrowNum:
           Wtilde.pop()
           Wtilde.pop()

        if len(Wtilde) == WrowNum :
            U, d, Vt = np.linalg.svd(Wtilde) # U가 왜 20 * 20이 나오지..?

            U_hat = U[:, 0:2] # U의 앞의 3열만 봐야함 -> 근데 L, R 2열 밖에 없어서 2열만!
#            U_hat[:, 0:3] = U[:, 0:3]

            D = np.zeros((PointNum, PointNum)) # d = 고유값 리스트이므로 대각행렬 D로 만들어줌
            for i in range(len(d)): # 시그마의 앞 3열, 3행만 봄 -> 지금 2 * 2
                D[i][i] = d[i]

#            Vt_hat[0:3, :] = Vt # Vt의 앞 3행만 봄 -> 지금 2행만
            Vt_hat = Vt[0:2, :]
            #R_hat = U_hat*D
            #R_hat = U * D**(1/2)
            R_hat = np.dot(U_hat, pow(D, 1/2)) # WrowNum x 3(2)
            S_hat = np.dot(pow(D, 1/2), Vt_hat) # 3 x P(2 x 2)

            # 식 (16)으로 Q 구함 / 3차원으로 해야되서 열 3개로 맞춰야되나...?
            Q = [(1, 0), (0, 1)] # 부호 모름 
            # Q = [(0, 1), (1, 0)] # 부호 모름
            # 식 (15)로 R, S 구함
            R = np.dot(R_hat, Q) 
            S = np.dot(np.linalg.inv(Q), S_hat)
            # 괜찮으면, 첫 카메라 기준계와 세계 기준계를 np.dot(R, R0), np.dot(R0.T, S) 생성해서 나란히 하기

    
            # 얼굴 깊이값 구하기
            # 눈 트래킹된 박스 부분 3차원으로
            if len(detected_face) == 1:
                FaceMatrix = np.array((detected_face[0][2], detected_face[0][3]))
                FaceOrigin = (detected_face[0][0] + detected_face[0][2] / 2, detected_face[0][1] + detected_face[0][2] / 2, 0)
                #for detected_face[0]
                faceHalf = w / 2 # 적절히 계산
                Zn = faceHalf / 3 # 적절히 계산
                r = (Zn * Zn + pow(faceHalf, 2)) / (2 * Zn) # faceHalf는 얼굴 트래킹해서 찾기, Zn은 faceHalf이용해서 계산..?
                Z = Zn - r + abs(pow(r, 2) - pow(y, 2)) # y = 얼굴 중심이 원점인 기준으로의 y좌표(얼굴 가로 좌표)


            # phi1이랑 R 크기 어떻게 맞추나..?
            Rtemp[0:4,:] = R[0:4,:]

            Rmatrix = np.dot(phi1, Rtemp) # phi1 : 3x4, Rtemp: 4 x 2=> 3 x 4이 되야하나
#            frame = cv2.warpAffine(frame, Rmatrix, (cols, rows)) # 3차원으로 회전해야 함..

        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        fps = 1 / sec
        str = "FPS : %0.1f" % fps
        cv2.putText(frame, str, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        #out.write(frame) # 이미지 파일로 저장
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
#out.release() # 비디오 저장 파일 종료 
cv2.destroyAllWindows()

