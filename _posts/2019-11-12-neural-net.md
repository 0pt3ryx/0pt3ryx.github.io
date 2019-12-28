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


# 출력층

## Softmax 함수

### 수식

### 간단 구현

~~~ python
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y
~~~


# 신경망 구현 예시

~~~ python
def init_network():
	with open('sample_weight.pkl', 'rb') as file:
		network = pickle.load(file)

	return network


def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)


data, label = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for idx in range(0, len(data), batch_size):
	x_batch = x[idx: idx+batch_size]
	y_batch = predict(network, x_batch)
	probability = np.argmax(y_batch, axis=1)
	accuracy_cnt += np.sum(probability == label[idx: idx+batch_size])

print('Accuracy: {0}'.format(str(float(accuracy_cnt) / len(x))))
~~~

[^ref-1]: 사이토 고키, Deep Learning from Scratch 밑바닥부터 시작하는 딥러닝, 한빛미디어