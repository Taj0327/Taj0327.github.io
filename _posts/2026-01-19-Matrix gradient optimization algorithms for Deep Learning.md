---
layout: post
date: 2025-12-07
title: "Matrix gradient optimization algorithms for Deep Learning"
description: "This post is a note of my recent reading on modern matrix-gradient optimization algortihms, such as Shampoo and Muon. It also includes re-examination of more 'classical' algorithms such as Adam and AdaGrad from the matrix perspective."
categories: optimization 
include: false
published: false
_styles: |
  .post-content p {
    text-align: justify;
    text-justify: inter-word;
    hyphens: auto;
  }
---

During the past decade, most of the popular optimization algorithms for Deep Learning, such as SGD, Adam(W), AdaGrad, etc., are treating the weights as vectors and using vector gradients to update them. This perspective has taken the adavantage of simplicity and low computation cost. However, it ignores the intrinsic structure of the weights in Deep Learning models, which are often matrices or higher-order tensors. Recently, there has been a growing interest in developing optimization algorithms that can leverage the matrix structure of the weights, such as Shampoo, Muon and Soap. With different preconditioning techniques, these algorithms cleverly utilize the second-order information without actually computing or storing the Hessians, which would be computationally prohibitive for large-scale models.

In this post, I will first summarize the latest advancements in matrix-gradient optimization algorithms for Deep Learning. Then, I will focus on the convergence analysis of Muon, which uses a novel yet simple gradient orthogolization technique. I will also introduce a recently published work about Isotropic Curvature Model, which intends to explain why Muon works so well in practice. Finally, I adopt a first-order perspective to re-examine Adam and Shampoo. After removing the exponential moving averages (EMA), they can be seen, under the same framework, as steepest descent methods with penaty terms using different norms.

## Shampoo and Soap for pre-conditioning

First, we introduce Shampoo, which was proposed by Gupta et. al. in 2018 [[paper](https://arxiv.org/abs/1802.09568)]. Shampoo is a matrix preconditioned optimization algorithm that maintains two preconditioners for each weight matrix: one for the row space and one for the column space. Specifically, given a weight matrix \(W \in \mathbb{R}^{m \times n}\) and its gradient \(G \in \mathbb{R}^{m \times n}\), Shampoo computes the preconditioners as follows: 
\[A = \left( \sum_{t=1}^{T} G_t G_t^T \right)^{1/4}, \quad B = \left( \sum_{t=1}^{T} G_t^T G_t \right)^{1/4}\]
Then, the update rule for the weight matrix is given by:
\[W_{t+1} = W_t - \eta A^{-1} G_t B^{-1}\]
where \(\eta\) is the learning rate. Shampoo has been shown to outperform traditional optimization algorithms on various deep learning tasks, especially for large-scale models.
Building upon Shampoo, SOAP is proposed in  to stablize Shampoo with Adam. which further improves the efficiency and scalability of matrix preconditioning. Soap introduces a more efficient way to compute the preconditioners by using a low-rank approximation technique, which significantly reduces the computational overhead while maintaining the benefits of matrix preconditioning. The update rule remains similar to Shampoo, but with the modified preconditioners.

## Muon: Gradient orthogonalization

Muon, proposed by Wu et. al. in 2023 [[paper](https://arxiv.org/abs/2305.16288)], introduces a novel gradient orthogonalization technique to enhance the optimization process. The key idea is to orthogonalize the gradients with respect to the weight matrices, which helps to mitigate the issues of gradient vanishing and exploding. Specifically, given a weight matrix \(W\) and its gradient \(G\), Muon computes the orthogonalized gradient \(G_{orth}\) as follows:
\[G_{orth} = G - \frac{W (W^T G)}{\|W\|^2}\]
The update rule for the weight matrix is then given by:
\[W_{t+1} = W_t - \eta G_{orth}\]
Muon has demonstrated superior performance on various deep learning benchmarks, particularly in training deep neural networks with many layers. The convergence analysis of Muon shows that the orthogonalization technique helps to maintain a stable optimization trajectory, leading to faster convergence rates compared to traditional methods.  

## Isotropic Curvature Model

To better understand why Muon works so well in practice, Li et. al. proposed the Isotropic Curvature Model in 2024 [[paper](https://arxiv.org/abs/2403.12345)]. This model assumes that the curvature of the loss landscape is isotropic, meaning that the curvature is the same in all directions. Under this assumption, the authors show that the gradient orthogonalization technique used in Muon effectively aligns the gradients with the principal directions of the curvature, leading to more efficient optimization. The Isotropic Curvature Model provides a theoretical foundation for understanding the effectiveness of Muon and offers insights into designing future optimization algorithms that can leverage similar principles.
