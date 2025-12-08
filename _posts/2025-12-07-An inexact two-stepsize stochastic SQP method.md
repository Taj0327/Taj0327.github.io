---
layout: post
date: 2025-12-07
title: "An inexact two-stepsize stochastic SQP method"
description: "This post introduces an inexact two-stepsize stochastic sequential quadratic programming (SQP) method for solving constrained optimization problems with noisy objective functions and deterministic constraints."
categories: optimization 
_styles: |
  .post-content p {
    text-align: justify;
    text-justify: inter-word;
    hyphens: auto;
  }
---

{: .justified}

This blog post introduces an inexact two-stepsize stochastic SQP method for nonlinear optimization problems with rank deficient constraint Jacobians. The problem is formulated as follows:

$$
\min_{x \in \mathbb{R}^n} f(x) = \mathbb{E}[F(x,\omega)] \quad \text{s.t.} \quad c(x) = 0,
$$

where $\omega$ is a random variable associated with probability space $(\Omega, \mathcal{F}, \mathcal{P})$ and $c(x)$ represents deterministic equality constraints. We assume that $f$ and $c$ are differentiable and have Lipschitz continuous gradients. Also, we assume that the constraint Jacobian $J_k = \nabla c(x)^T$ may be rank deficient, i.e., the Linear Independence Constraint Qualification (LICQ) may not hold. For the stochasticity part, we only have access to an unbiased estimate of the objetive gradient $\nabla f(x_k)$ at each iteration $k$. Specifically, we can obtain a stochastic gradient estimate $g_k$ satisfying $\mathbb{E}[g_k] = \nabla f(x_k)$ and $$\mathbb{E}[\|g_k - \nabla f(x_k)\|_2^2] \leq M.$$

### Algorithm Description

Traditionally, SQP methods solve a sequence of quadratic programming (QP) subproblems to generate search directions $d_k$, and update the iterates via $x_{k+1} = x_k + \alpha_k d_k$, where $\alpha_k$ is some stepsize. The nonlinear CG method is often used to solve the QP subproblems approximately. However, when the constraint Jacobian is rank deficient, the standard SQP method may fail to find a feasible direction. To address this problem, we use the Byrd-Omojokun framework to decompose the step direction $d_k$ into a normal component $v_k$ and a tangential component $u_k$. The normal step $v_k$ is computed via a trust-region subproblem, while the tangential step $u_k$ is obtained by solving a quadratic programming (QP) subproblem with iterative linear solvers (such as MINRES or CG).

More specifically, the normal step $v_k$ is computed by solving the following trust-region subproblem:

$$
\min_{v \in \mathbb{R}^n} \|c(x_k) + J_k^T v\|^2 \quad \text{s.t.} \quad \|v\| \leq \omega \|J_k^Tc_k\|_2,
$$

where $\omega$ is a positive constant. Taking a symmetric matrix $H_k\in\mathbb{S}^n$ that satisfies . The tangential step $u_k$ is computed by solving the following Newton-type system:

$$
\begin{bmatrix}
    H_k & J_k^T \\
    J_k & 0
\end{bmatrix} \begin{bmatrix}
    u_k \\ y_{k}
\end{bmatrix} = \begin{bmatrix}
    -g_k + H_k v_k \\ 0
\end{bmatrix}.
$$

When the subproblem is solved exactly, the solution $u_k$ lies in the null space of $J_k$, i.e., $J_ku_k = 0$, ensuring that the tangential step does not affect the feasibility achieved by the normal step. However, since we only have access to a stochastic gradient estimate $g_k$, we allow inexact solutions to the Newton-type system. Specifically, we use a residual stopping criterion to terminate the iterative solver:

$$
\|r_k\| \leq \gamma_r \beta_k \quad \& \quad \|\rho_k\| \leq \gamma_{\rho} \beta_k,
$$

where $(r_k, \rho_k)$ is the residual pair of the system:

$$
\begin{bmatrix}
    r_k \\ \rho_k
\end{bmatrix} := \begin{bmatrix}
    H_k & J_k^T \\
    J_k & 0
\end{bmatrix} \begin{bmatrix}
    u_k \\ y_{k}
\end{bmatrix} + \begin{bmatrix}
    g_k + H_k v_k \\ 0
\end{bmatrix}.
$$

Unlike the common way of simply adding $u_k$ with $v_k$, we use a rescaling stepsize $\beta_k$ to control the stochastic variance introduced by the tangential step. The overall step is given by $$d_k = \beta_k u_k + v_k$$. The stepsize $\beta_k$ acts as a variance reduction technique, which helps to stabilize the iterates and improve convergence. The stepsize $\alpha_k$ is chosen from a pre-defined interval related to the problem constants. A generic algorithm framework can be summarized as follows:

Initialize point $x_0 \in \mathbb{R}^n$.

$\textbf{For}$ $k = 0,1,2,\dots,K$ $\textbf{do}$:

1. Compute the normal step $v_k$ by solving the trust-region subproblem.
2. Compute the stochastic gradient estimate $g_k$.
3. Compute the tangential step $u_k$ by solving the Newton-type system.
4. Choose $\beta_k > 0$ and set $d_k = \beta_k u_k + v_k$.
5. Choose stepsize $\alpha_k > 0$.
6. Update $x_{k+1} = x_k + \alpha_k d_k$.

$\textbf{End For}$

### Complexity Analysis

We use a $l_2$-norm merit function to measure the progress of the algorithm:

$$
\phi(x, \tau) = \tau f(x) + \|c(x)\|_2.
$$

The merit function balances the objetive and constraint violation by the merit parameter $\tau > 0$. Since no LICQ is assumed, we use the projection matrix $Z_k$ onto the null space of $J_k$ to define stationarity.

A point $x$ is called an $\varepsilon_L$-stationary point if $$\|Z_k^T \nabla f(x)\|_2 \leq \varepsilon_L$$. Also, a point $x$ is called an $\varepsilon_c$-feasible point if $$\|J_k^Tc_k(x)\|_2 \leq \varepsilon_c$$. This is an extension of the traditional KKT conditions for constrained optimization problems. If LICQ holds, then $Z_k$ becomes full rank and $$\|J_k^Tc_k\|_2$$ is equivalent to $$\|c(x)\|_2$$, which recovers the standard definitions of stationarity and feasiblity.

We analyze the decrease of $\phi$ through a first-order approximation near $x$:

$$
l(x, \tau, g, d) = \tau (f(x) + g^T d) + \|c(x) + J^T d\|_2.
$$

$l$ serves as a surrogate model for $\phi$ at each iteration. By carefully bounding the expected decrease of $l$ ($\Delta l$) and choosing appropriate parameters, we can establish the convergence properties of the proposed method under three different cases:

1. When no constraint qualification is assumed and the merit parameter sequence $\{\tau_k\}$ is lower bounded by a positive constant $\tau_{\min} > 0$.
2. When LICQ holds at every iteration. (This is a sufficient condiiton for $\tau_{\min} > 0$.)
3. When the merit parameter sequence $\{\tau_k\}$ vanishes to zero.

For the first two cases, an important lemma bounds the decrease in the merit function:

**Lemma:** Under suitable assumptions and parameter choice, if $\tau_k \geq \tau_{\min} > 0$ and $\alpha_k\leq 1$, we have

$$
\begin{aligned}
    \phi(x_k+\alpha_k d_k, \tau_{\min}) - \phi(x_k, \tau_{\min}) \leq -\alpha_k \Delta l(x_k, \tau_{\min}, \nabla f(x_k), d_k^\text{true}) +  \\ \frac{\alpha_k^2\textcolor{red}{\beta_k^2}}{2}(\tau_{\min} L + \Gamma) \|u_k\|_2^2 + \frac{\alpha_k^2}{2}(\tau_{\min} L + \Gamma)\|v_k\|_2^2  +\alpha_k \tau_{\min} \nabla f(x_k)^T\textcolor{red}{(d_k - d_k^\text{true})}.
\end{aligned}
$$

We can see that the rescaling stepsize $\beta_k$ appears in the second term on the right-hand side, which captures the variance introduced by the tangential step. Also, the term $d_k - d_k^\text{true} = \beta_k(u_k - u_k^\text{true})$ penalizes the error between the stochastic tangential step and its true counterpart. By choosing $\beta_k$ appropriately, we can control these variance terms and ensure sufficient decrease in the merit function.

Based on this lemma, taking expectation and summing over $k=0,1,\dots,K-1$, we can derive the following complexity results:

**Theorem:** Under suitable assumptions and parameter choices, the proposed algorithm achieves the following complexity bounds:

1. If no constraint qualification is assumed and $\tau_k \geq \tau_{\min} > 0$, it needs at most $\mathcal{O}(\varepsilon_c^{-2})$ iterations to find an $\varepsilon_c$-feasible point and $\mathcal{O}(\varepsilon_L^{-4})$ iterations to find an $\varepsilon_L$-stationary point.
2. If LICQ holds at every iteration, it needs at most $\mathcal{O}(\varepsilon_c^{-1})$ iterations to find an $\varepsilon_c$-feasible point and $\mathcal{O}(\varepsilon_L^{-4})$ iterations to find an $\varepsilon_L$-stationary point.

The first worst-case infeasibility complexity result is the **first-known** result for stochastic SQP methods without any constraint qualification. The second result matches the best-known complexity for stochastic SQP methods under LICQ. For the KKT complexity, our result matches the best-known complexity for stochastic first-order methods for unconstrained optimization.

For the third case where $\tau_k \to 0$, since the merit function ignores the objective function asymptotically, we can only guarantee convergence to some feasible point or stationary infeasible point:

**Theorem:** Under suitable assumptions and parameter choices, if $\tau_k \to 0$ as $k \to \infty$, 

$$\liminf_{k \to \infty} \|J_k^T c_k\|_2 = 0.$$

### Numerical Experiments

We conduct numerical experiments on a total of 110 test problems from the CUTEst library to evaluate the performance of the proposed method. We compare our inexact two-stepsize stochastic SQP method (ITSQP) with a baseline stochastic SQP method (SSQP) that does not use the rescaling stepsize $\beta_k$, as well as a step-lengthening SSQPL method. Since singularity is detected in more than half of the problems, we use the step decomposition strategy for all three algorithms.

Multiple levels of stochasticity (noise follows $$\mathcal{N}(0, \epsilon_N I)$$, $$\epsilon_N\in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1} \}$$) are injected into the gradient estimates to simulate the stochastic gradient. To ``tune" the algorithms, we tested 5 different fixed scaling stepsizes $$\beta \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1},1\}$$ under 15 different seeds. Therefore, a total of 440(= 110$\times$4) problems are tested, and for each problem, every algorithm is run 75(= 5$\times$15) times with different stepsizes. For the stepsize $\alpha_k$, we use the default stepsize choosing scheme for SSQP and a step-lengthening stepsize for SSQPL.

For each run of the algorithm, we give a budget of 10,000 iterations and report the best iterate with the following scheme: for each iterate k, if $$\|c_k\|_\infty \leq 10^{-6}$$, we treat it as a feasible point. We pick the feasible iterate with lowest KKT error $$\|\nabla f(x_k) + J_k^Ty_k\|_\infty$$. If no feasible point is found, we pick the iterate with the lowest infeasibility error $$\|c_k\|_\infty$$ and report its KKT error as well. The KKT error and infeasibility error of the best iterates are averaged over the 15 seeds for each problem.

The figures contain the KKT errors and infeasibility errors of three algorithms tested under different noise levels. For each of the 440 problems, we compared the average infeasibility errors and KKT errors over five different stepsizes, and chose the one stepsize that yields the most best results to plot the figures. It is $\alpha =1$ for SSQP and SSQPL, and $\beta_k = 10^{-3}$ for ITSQP.

The results show that ITSQP outperforms both SSQP and SSQPL in terms of feasibility. The advantage becomes increasingly pronounced as the noise level increases, highlighting the superior convergence rate of our two-stepsize algorithm. The SSQP and SSQPL methods compute the merit parameter sequence $\tau_k$ and choose the stepsize based on $\tau_k$, so they achieve better KKT convergence. We argue that their lower KKT errors are obtained in the cost of computational resources. The higher KKT errors may also be attributed to the more pessimistic stepsize $\alpha_k$ we use in our algorithm.

<p align="center">
  <img
    src="/assets/img/posts/complexity results.png"
    width="900"
    alt="My figure"
  />
</p>

For more details, please check my master essay <a href="/assets/pdf/Masters_Essay_Aoji_Tang.pdf" target="_blank" rel="noopener noreferrer"> here </a>.

