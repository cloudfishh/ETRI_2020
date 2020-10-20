해당 코드는 python 3.7.3 기준으로 작성되었습니다.

NaN을 인공적으로 주입하는 코드가 함께 작성되어 있으며,
테스트 대상으로 했던 label_data.csv의 68181c16 세대에 대한 injected 파일을 동봉합니다. (68181c16_injected.csv)
mask_inj는 각각 0: valid data, 1: original NaN, 2: injected NaN, 3: normal point, 4: accumulated point를 의미합니다. 


calendar 폴더를 데이터가 포함된 디렉토리에 함께 넣어주십시오.
imputation.py에서 테스트할 지역/아파트/세대를 입력하면 테스트가 진행됩니다.

imputation.py
	line 16: 데이터가 포함되어 있는 디렉토리 경로를 입력하십시오.
	line 17: 테스트할 지역을 입력하십시오. (광주, 나주, 대전, 서울, 인천, label)
	line 18: 테스트할 아파트의 법정동 코드를 입력하십시오.
	line 19: 세대의 식별 코드를 입력하십시오.

코드 시행 시 detection 결과가 confusion matrix로 저장되며,
imputation 결과로 accuracy가 출력됩니다.