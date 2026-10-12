YOLO 형식 자동 박스 라벨 데이터
images/와 labels/의 하위 경로와 파일명이 서로 대응합니다.
라벨: class_id x_center y_center width height (0~1 정규화)
20+6은 하나의 클래스로, 이미지마다 박스 1개입니다.
박스는 흰 여백을 제외한 전경 범위로 자동 계산했습니다.
함께 찍힌 문자/선/화살표도 포함될 수 있어 원본 69장의
box_previews와 source_boxes.json을 검수해야 합니다.
증강 좌표는 원본 박스의 90도 회전, 크기 및 이동을 반영합니다.
split은 미지정입니다. index.json의 source_id와 실제 중복/겹침을
기준으로 나누고, train에만 해당 원본의 증강본을 사용하세요.
원본과 증강본을 임의로 섞어 val/test에 넣으면 평가가 부풀려집니다.
학습은 실행하지 않았습니다.
