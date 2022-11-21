# 2022 카타르 월드컵 승부 예측: 프로젝트 개요

본 프로젝트는 90년대 이후 국제 경기 전적, 최근 경기 결과, 잠재능력 등을 바탕으로 2022 카타르 월드컵의 결과를 예측하는 것을 목표로 한다.

## Resources Used

* Python Version: 3.7
* Packages: Pandas, NumPy, Sklearn, Tensorflow, and Seaborn.
* Data: 
  * [international_matches.csv](https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022) - This dataset provides a complete overview of all international soccer matches played since the 90s. On top of that, the strength of each team is provided by incorporating actual FIFA rankings as well as player strengths based on the EA Sport FIFA video game.
  * [players_22.csv](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) - The datasets provided include the player data for FIFA 22 Career Mode.

## 데이터 사전준비 및 데이터셋 만들기

* 사용될 데이터셋 [international_matches.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/international_matches.csv) 과 [players_22.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/players_22.csv) 모두 분석과 머신러닝 모델의 훈련 데이터셋으로 사용하기 위해 만들어졌다. 출전하지 않는 팀에 대한 데이터가 제거되었고 Nan 값 역시 대체된 상태이다.

* 데이터셋 [international_matches.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/international_matches.csv)으로부터 훈련 데이터셋인[training.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/training.csv)과 추론 데이터셋인 [last_team_scores.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/last_team_scores.csv)을 생성한다. [training.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/training.csv)은 상대하는 팀의 이름, FIFA 랭킹, 수치화된 양팀의 수비, 미드필드, 공격 레이팅을 포함한다. 반면에, 추론 데이터셋은 가장 최신의 FIFA 국제전 경기의 데이터를 포함한다.

## 탐색적 데이터 분석(EDA, Exploratory Data Analysis)

데이터셋 [international_matches.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/international_matches.csv) 와 [players_22.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/players_22.csv)으로부터 노트북 [QATAR22_EDA+Data_Preparation.ipynb](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/QATAR22_EDA%2BData_Preparation.ipynb) 와 [Getting_Squads_Stats.ipynb](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/Getting_Squads_Stats.ipynb)은 하단의 질문에 대한 답을 제공한다. 하단의 질문들은 통계에 따른 월드컵 우승후보를 찾을 수 있는 아이디어를 제공한다.

* 어느 팀이 가장 좋은 공격을 보유했는가?

![download-12](https://user-images.githubusercontent.com/60159274/193368513-18266a41-cef3-4dac-9273-dfecc0357b3e.png)

* 어느 팀이 가장 좋은 수비를 보유했는가?

![download-8](https://user-images.githubusercontent.com/60159274/193368518-889e1672-d759-4e80-85e6-6de64f6f3e1c.png)

* 어느 팀이 가장 좋은 중원을 보유했는가?

![download-11](https://user-images.githubusercontent.com/60159274/193368515-2a046f68-61a2-421d-9e74-b0420bd452e9.png)

* 어느 팀이 가장 높은 승률을 보유했는가?

![download-6](https://user-images.githubusercontent.com/60159274/193368516-68e21bf6-bc91-4759-80d9-767271dc0636.png)

* 어느 선수가 2022 카타르월드컵 최고의 선수일까?

![download-2](https://user-images.githubusercontent.com/60159274/193378980-b2302754-6514-449d-b890-cce0a716a519.png)

* 가장 유망한 팀은 어디인가?

![download-5](https://user-images.githubusercontent.com/60159274/193368544-11f6f51f-2a2d-4812-af89-4eed5e71763d.png)

* 로컬 팀의 이점이 있는가?

이 질문은 매우 중요하다. 아래의 그래프는 홈팀이 홈에서 열리는 경기의 50% 이상을 승리했음을 보여준다. 홈 구장의 익숙함, 원정팀의 이동 거리, 우리 영역이라는 안도감, 홈팀 팬들의 열띤 응원 등 다양한 이유가 있을 수 있다. 예를 들어 콜롬비아 대표팀이 브라질을 상대하기 위해 마랑카나 스타디움을 방문하면 보통은 지거나 비기는 경기를 한다. 하지만 콜롬비아의 홈 구장인 메트로폴리나토 스타디움에서 브라질을 상대하면 비기거나 이기는 양상을 보인다. 이러한 이유에서, 승부결과를 예측할 수 있는 기계학습 모델을 구축하기 위해서는 홈 팀과 원정 팀을 반드시 구분해야 한다.

![download-7](https://user-images.githubusercontent.com/60159274/193368561-dd1398c8-dcad-4575-b3aa-2f4a30719444.png)

## 모델링과 튜닝

노트북 [Modeling+Tuning.ipynb](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/Modeling%2BTuning.ipynb)은 월드컵 경기의 결과를 예측하는 기계 학습 모델을 훈련시키는 것을 목적으로 한다. 해당 노트북은 조별 단계를 예측하기 위한 기계 학습 모델 하나와, 녹아웃 단계를 예측하기 위한 모델 하나를 선정한다. 차이점은 조별 단계의 경기들은 3-Way(승,무,패)의 결과를 갖지만 녹아웃 스테이지(16강 이상의 경기)는 승리와 패배만 존재한다. 각 단계별 최고의 모델은:

* Random Forest
* Ada Boost Classifier
* XGB Boost
* Neural Networks

XGB Boost 모델이 모든 단계에서 최고의 성능을 나타냈다. 그러므로 쉬운 추론을 위해 해당 모델을 튜닝, 검증 후 파이프라인으로 구축하였다.

* 튜닝, 검증된 그룹 스테이지 모델의 오차 행렬(Confusion Matrix)

![download-10](https://user-images.githubusercontent.com/60159274/193368594-3d6f69a8-cc6c-456c-9408-a2ebc1f72ee1.png)

* 튜닝, 검증된 녹아웃 스테이지 모델의 오차 행렬(Confusion Matrix)

![download-9](https://user-images.githubusercontent.com/60159274/193368596-cbd0a492-7399-49af-be28-bd6c4a014694.png)

## 결과 예측

마지막으로, 노트북 [Predictions.ipynb](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/Predictions.ipynb)은 추론 데이터셋과 훈련된 모델을 이용하여 월드컵 경기의 승자를 예측하고 궁극적으로 월드컵의 우승자를 예측할 수 있다. 각 월드컵 경기에서 홈팀을 선택하려면 각 팀의 잠재력을 제공하는 데이터 세트 [squad_stats.csv](https://github.com/davidcamilo0710/QATAR_2022_Prediction/blob/master/data/squad_stats.csv)를 사용하므로 더 중요한 잠재력을 가진 팀이 홈팀이 될 것이다.
