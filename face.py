import cv2

# 1. 사용할 얼굴 검출 모델(Haar Cascade) 로드
# OpenCV에 내장된 정면 얼굴 인식 데이터를 불러옵니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 웹캠 연결 (0번은 기본 내장 카메라)
cap = cv2.VideoCapture(0)

print("얼굴 검출을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 속도와 정확도를 위해 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. 얼굴 검출 수행
    # scaleFactor: 이미지 크기 감소 비율 (1.1 ~ 1.3 권장)
    # minNeighbors: 얼굴 후보 사각형이 유지되어야 하는 최소 개수 (3 ~ 6 권장)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # 4. 검출된 얼굴에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 결과 화면 표시
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()