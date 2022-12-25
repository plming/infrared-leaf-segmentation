# infrared-leaf-segmentation

## Introduction

식물의 잎을 촬영한 적외선(IR) 이미지에서 잎 영역을 분할(segmentation)하는 프로젝트입니다.

## Convention

* 컬러 이미지의 채널 순서는 RGB가 아닌 BGR로 한다.(ex. Red == [0, 0, 255])
* opencv-python 패키지 버전 변경하지 않는다.(Pycharm의 suggestion 기능이 작동하지 않음)

## Idea notes

* Watershed
* Graph cut(GrabCut)
* Neural Network