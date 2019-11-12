---
layout: post
title: "신경망"
author: "0pt3ryx"
description: "신경망 내용 정리 (Deep Learning from Scratch)"
categories: [deep-learning]
tags: [machine-learning, deep-learning, python, deep-learning-from-scratch, activation-function]
redirect_from:
  - /2019/11/12/
---

> 책 `Deep Learning from Scratch`[^ref-1]를 읽고 그 내용을 정리한 포스트이다.

* Kramdown table of contents
{:toc .toc}

# 신경망의 예시


# 활성화 함수(activation function)

입력 신호의 총합이 활성화를 일으키는지를 정하는 역할을 하는 함수를 활성화 함수라고 한다.

## 시그모이드 함수(sigmoid function)

/수식/
0에서 1사이의 실수 값을 가진다. 비선형 함수에 해당한다.

### 간단 구현

~~~ python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
~~~

## ReLU(Rectified Linear Unit) 함수

/수식/
0이상의 값은 같은 값을, 0미만의 값은 0을 출력하는 활성화 함수

### 간단 구현

~~~ python
import numpy as np


def relu(x):
    return np.maximum(0, x)
~~~

[^ref-1]: 사이토 고키, Deep Learning from Scratch 밑바닥부터 시작하는 딥러닝, 한빛미디어