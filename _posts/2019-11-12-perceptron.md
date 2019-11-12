---
layout: post
title: "퍼셉트론"
author: "0pt3ryx"
description: "퍼셉트론 내용 정리 (Deep Learning from Scratch)"
categories: [deep-learning]
tags: [machine-learning, deep-learning, python, deep-learning-from-scratch, perceptron]
redirect_from:
  - /2019/11/12/
---

> 책 `Deep Learning from Scratch`[^ref-1]를 읽고 그 내용을 정리한 포스트이다.

* Kramdown table of contents
{:toc .toc}

# Single-Layered Perceptron

퍼셉트론(단순 퍼셉트론)은 다수의 신호를 입력으로 받아 하나의 신호를 출력한다.

## 구현

~~~ python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
~~~

## 한계

단층 퍼셉트론(single-layer perceptron)으로는 XOR을 구현할 수 없다. 즉, 퍼셉트론 하나만으로는 직선 하나로 나눈 영역만 표현할 수 있으나 XOR은 비선형 영역이기 때문에 구현할 수 없다.

# Multi-Layered Perceptron

여러 단층 퍼셉트론을 층을 쌓으면 다층 퍼셉트론이 된다. 다층 퍼셉트론은 XOR을 구현할 수 있다. 즉, 다층 퍼센트론은 비선형적 표현이 가능하다.

## 구현

~~~ python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
~~~

[^ref-1]: 사이토 고키, Deep Learning from Scratch 밑바닥부터 시작하는 딥러닝, 한빛미디어