---
layout: post
title: Constrained Optimization
date: 2025-09-26 22:36:00
description: Some notes on constrained optimization
tags: formatting charts
categories: optimization
chart:
  plotly: true
---

Welcome! This post contains some notes on constrained optimization, mainly based on the book [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) by Jorge Nocedal and Stephen Wright.

## Introduction

Constrained optimization problems are optimization problems that include constraints on the variables. A general constrained optimization problem can be formulated as follows:
\[\begin{aligned}
& \min\_{x \in \mathbb{R}^n} f(x) \\
& \text{subject to} \\
& g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}\]
where \(f(x)\) is the objective function, \(g_i(x)\) are the inequality constraints, and \(h_j(x)\) are the equality constraints.
