---
layout: post
date: 2026-10-28
title: "A two-phase SQP method under a constraint PL condition"
description: "A two-phase SQP method for equality constraints, with feasibility guarantees under a constraint PL condition."
categories: optimization
published: true
---

*Draft based on ongoing joint work with Michael J. O'Neill.*

## Why two phases?

In equality-constrained optimization, we want to minimize an objective while satisfying a set of constraints:

$$
\min_{x\in\mathbb{R}^n} f(x)
\qquad\text{subject to}\qquad c(x)=0.
$$

There are two goals: make the constraint violation small and improve the objective along the constraints. These goals can require different levels of accuracy. For example, a physical constraint may need to be satisfied much more accurately than the desired first-order optimality tolerance.

In [our earlier stochastic SQP post]({{ '/blog/2025/An-inexact-two-stepsize-stochastic-SQP-method/' | relative_url }}), we discussed splitting an SQP direction into normal and tangential components. The two-phase method uses the same idea, but changes the balance between the components according to the current level of constraint violation. A key ingredient is a **constraint Polyak–Łojasiewicz (PL) condition**: it allows us to analyze progress toward actual feasibility even when the constraint Jacobian is rank deficient.

## Step update in both phases

At iteration $k$, let $c_k=c(x_k)$ and $J_k=\nabla c(x_k)^T$. We compute a **normal step** $v_k$ to reduce the linearized constraint violation, and a **tangential step** $u_k$ that uses objective-gradient information while satisfying $J_ku_k=0$. The full step update has the form

$$
d_k=v_k+\beta_k u_k,
\qquad x_{k+1}=x_k+\alpha_k d_k.
$$

The factor $\beta_k$ balances the two components, while stepsize $\alpha_k$ controls how far we move along $d_k$. Since $J_ku_k=0$, the tangential component does not change the *linearized* constraint residual: $J_kd_k=J_kv_k$. For nonlinear constraints, however, a large tangential step can still affect the actual constraint violation through higher-order terms.

The deterministic method switches between two types of steps:

- **V-steps** focus on feasibility. They retain the tangential component but scale it so that $\|\beta_k u_k\|\leq\rho\|v_k\|$, where $\rho>0$ is a chosen parameter. The step length is checked against both a merit function and a constraint-decrease condition.
- **U-steps** focus on optimality once the current feasibility threshold is met. They use the unscaled combined direction $v_k+u_k$ and a merit-function line search. A U-step can increase constraint violation, in which case the method can return to V-steps.

For instance, when $u_k\neq0$, the deterministic V-step scaling is

$$
\beta_k=\min\left\{1,\frac{\rho\|v_k\|}{\|u_k\|}\right\}.
$$

Thus, a feasibility step can still use objective information without allowing the tangential component to overwhelm the normal step. The phase threshold can also be tightened adaptively when more feasibility is needed to assess optimality in the algorithm.

## The constraint PL condition

For the analysis, define the least-squares feasibility function

$$
\psi(x)=\frac12\|c(x)\|^2,
\qquad \nabla\psi(x)=J(x)^Tc(x).
$$

A small $\|J(x)^Tc(x)\|$ means that we are nearly stationary for the feasibility function, but it does not necessarily mean that $c(x)$ is small. To connect these two measures, we assume that the constraint system is feasible and that, on the region used in the analysis, there is a constant $\sigma_J>0$ such that

$$
\|J(x)^Tc(x)\|\geq\sigma_J\|c(x)\|.
$$

The condition says that the feasibility gradient cannot be too small relative to the constraint residual. It is equivalent to a PL inequality for $\psi$ when the minimum of $\psi$ is zero. This rules out infeasible stationary points of the feasibility function within the region where the condition holds.

Constraint PL is weaker than **uniform LICQ**: it controls the current residual direction rather than every direction in constraint space. For example, $c(x)=(x,x)^T$ has redundant constraints and a rank-deficient Jacobian, but the inequality holds with $\sigma_J=\sqrt{2}$. Thus, the condition can support a feasibility guarantee even when the constraint gradients are not linearly independent.

Without constraint PL, a small feasibility gradient alone need not imply feasibility, as discussed in the [earlier post]({{ '/blog/2025/An-inexact-two-stepsize-stochastic-SQP-method/' | relative_url }}).

## What the deterministic analysis says

Under constraint PL and the paper's smoothness and subproblem assumptions, each consecutive block of V-steps reduces the constraint violation geometrically. For a fixed optimality tolerance, the resulting upper bound on a restoration block grows logarithmically as the feasibility tolerance is tightened. A later U-step may increase the violation, so this is a guarantee for each restoration block, not a claim that every subsequent iterate remains feasible.

The deterministic analysis also counts the work needed to meet both a feasibility tolerance $\varepsilon_c$ and a first-order tolerance $\varepsilon_L$. For $0<\varepsilon_c,\varepsilon_L\leq1$, the draft gives the total iteration bound

$$
\mathcal O\!\left(\min\left\{
\max\{\varepsilon_c^{-1},\varepsilon_L^{-2}\},\;
\varepsilon_L^{-2}\log(1+\varepsilon_c^{-1})
\right\}\right).
$$

The two expressions are bounds for the same algorithm: one counts the total number of feasibility and optimality steps, while the other combines the number of optimality steps with the length of each restoration block. The logarithmic dependence can be especially useful when a much tighter feasibility tolerance is needed. The result concerns approximate feasibility and first-order stationarity; it does not guarantee a global minimum.

The same two-phase idea can also be used with stochastic objective gradients, while the constraints and their Jacobian remain deterministic. The stochastic analysis and numerical comparison are still being developed, so this post focuses on the deterministic result.
