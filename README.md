# Improvement performance of sales forecasting machine learning model by various data pool - focusing on clustering as way of partial pooling

경희대학교 소프트웨어융합캡스톤디자인
1. Walmart의 상품별 판매량 데이터를 활용해 수요 예측 모델을 구축하여 성능을 비교하는 M5 Accuracy Competition에서 우승한 모델인 DRFAM보다 더 나은 성능을 보이는 모델을 구축하고, 구축해야 하는 모델의 개수를 줄이기 위함이 목적이었다. 
2. 기존 모델은 트레이닝에 활용하는 데이터 풀을 구성할 때, 상품별 판매량 특성을 반영하지 않은 지점이나 지역과 같은 특성 또는 제품의 카테고리와 같은 분류체계를 활용해 데이터를 나누어 각 데이터 풀에 맞는 모델을 구축하였다. 해당 프로젝트는 이 부분에서 한계점을 발견하여 데이터 풀을 수요 예측에 크게 영향을 미치는 판매량 추이 데이터의 특성을 반영한 데이터 풀을 구성하고, 구성한 각 데이터 풀에 적합한 모델을 구축하여 DRFAM의 성능을 개선하고자 했다. 이 과정에서 데이터 풀을 구성하는 방법으로 time series k-mean를 선택하여 진행하였다. 또한 데이터의 레벨에 따라 클러스터링을 진행하여 partial pooling을 진행했던 데이터 레벨이 변화함에 따라 성능이 변화하는지 확인하였다.

# 과정
1. 데이터 전처리   
 (1) item level clustering(preprocessing_clustering_item_storedept.ipynb)
 - sales_train_evaluation.csv의 데이터 중 item_id가 같은 상품을 묶어 일별 판매량 평균 산출
 - 판매량 추이 데이터를 time seriese k-means를 통해 유사한 판매량 추이를 가지고 있는 item을 10개의 클러스터로 클러스터링(metric=euclidean)
 - 추후 모델 트레이닝 시 발생하는 메모리 에러 발생 방지를 위해 각 클러스터 내의 데이터 개수가 10,000개 이상인 클러스터(클러스터에 속하는 item의 개수가 1,000개 이상인 클러스터)를 선택하여 5개의 클러스터로 재클러스터링
 - 클러스터 레이블 중복 방지를 위해 재클러스터링한 레이블을 변경한 후, 각 아이템별로 해당하는 클러스터 레이블 넘버 컬럼을 결합
 - 기존 데이터셋에 item_id를 기준으로 클러스터 레이블 넘버 컬럼 결합
 2) store-department level clustering(preprocessing_clustering_item_storedept.ipynb)
 - sales_train_evaluation.csv의 데이터 중 store_id와 dept_id가 같은 상품을 묶어 일별 판매량 평균 산출(70개 항목 생성)
 - 판매량 추이 데이터를 time seriese k-means를 통해 유사한 판매량 추이를 가지고 있는 item을 10개의 클러스터로 클러스터링(metric=euclidean)
 - 기존 데이터셋에 각 store, department id별로 해당하는 클러스터 레이블 넘버 컬럼을 결합
 3) 모델 구축 시 활용할 feature 생성(preprocessing-cluster.ipynb)
 - 기존 DRFAM 모델 구축 시 진행했던 데이터 전처리 과정과 동일
 
2. TRAIN(1-1. recursive_deptstore_cluster10_TRAIN.ipynb, 1-1. recursive_item_clustering_TRAIN.ipynb, 2-3. nonrecursive_storedept_cluster_TRAIN.ipynb, 2-1. nonrecursive_item_clustering_TRAIN.ipynb)
- recursive와 non-recursive 모델을 각각 구축(item level: 14 *2 = 28개 모델, store-dept level: 10 * 2 = 20개 모델)
- 데이터 풀의 차이에 따른 성능 변화를 측정해야 하므로 모델의 파라미터는 기존과 동일하게 유지
- 기존 모델은 store id나 dept id에 따라 모델을 training을 진행했고, 본 과제에서는 cluster_id별로 모델 training 진행

3. PREDICT(1-3. recursive_cluster10_storedept_PREDICT.ipynb, 1-1. recursive_item_clustering14_PREDICT.ipynb, 2-3. nonrecursive_cluster10_storedept_PREDICT.ipynb, 2-1. nonrecursive_item_clustering14_PREDICT.ipynb)
- 구축한 recursive와 non-recursive 모델을 가져와 각 클러스터별로 미래 판매량 예측
- 데이터 풀의 차이에 따른 성능 변화를 측정해야 하므로 모델의 파라미터는 기존과 동일하게 유지
- 기존 모델은 store id나 dept id에 따라 판매량을 예측했고, 본 과제에서는 cluster_id별로 판매량 예측

4. AVERAGE(3-1. Final ensemble-clustering.ipynb, 3-1. Final ensemble-item-clustering.ipynb)
- recursive와 non-recursive 모델을 통해 예측한 일별 판매량을 평균을 취해 최종 판매량 예측치 산출
- 최종적으로 산출된 판매량 예측치로 WRMSSE 값을 구하여 성능 평가

# 결과
1) 소비액 데이터의 경우, 숙박비의 영향력이 증가하고, 음식비와 여행활동비의 비중이 감소한 부분에 주목해야 한다. 코로나19로 인해 사회적 거리두기를 시행하고 있고, 이에 따라 여행 활동이나 식당에서 음식을 먹는 관광객이 줄어들어 자연스레 총 소비액에 대한 비중이 감소했다고 볼 수 있다. 대신, 관광객들이 본인의 일행과 호텔이나 펜션과 같은 장소를 관광에 대한 대안으로 더 많이 찾으면서 여가 시간을 즐겼기 때문에 위와 같은 결과가 나왔다고 설명할 수 있다. 
2) 관광지 선택 이유의 경우, 데이터를 가공하는 방식을 바꾸고, 모델에 적용되는 파라미터를 변경해도 모델의 정확도가 0.2 주변에서 맴도는 것을 확인할 수 있었다. 결국, 본 분석에 사용된 관광객의 개인 특성 요소들만으로는 관광지를 선택하는 이유를 설명하기 어렵다고 판단할 수 있다. 
