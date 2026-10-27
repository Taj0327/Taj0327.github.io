---
layout: post
date: 2025-12-07
title: "An inexact two-stepsize stochastic SQP method"
description: "A short introduction to our stochastic SQP method for equality constraints with possibly rank-deficient Jacobians."
categories: optimization
_styles: |
  #markdown-content table {
    border-collapse: collapse;
    width: 100%;
  }
  #markdown-content th,
  #markdown-content td {
    border: 1px solid #cbd5e1;
    padding: 0.55rem 0.7rem;
    vertical-align: top;
  }
  html[data-theme="dark"] #markdown-content th,
  html[data-theme="dark"] #markdown-content td {
    border-color: #52525b;
  }
---

*Updated October 2026. This post summarizes [our April 2026 preprint](https://arxiv.org/abs/2604.14351), joint work with Michael J. O'Neill. The [earlier master's essay]({{ '/assets/pdf/Masters_Essay_Aoji_Tang.pdf' | relative_url }}) contains a previous version.*

## The problem

We study equality-constrained optimization with a stochastic objective:

$$
\min_{x\in\mathbb{R}^n} f(x)=\mathbb{E}[F(x,\xi)]
\qquad\text{subject to}\qquad c(x)=0.
$$

Here $c:\mathbb{R}^n\to\mathbb{R}^m$ is deterministic and $\xi$ is the random variable. At iteration $k$, let $c_k=c(x_k)$ and $J_k=\nabla c(x_k)^T\in\mathbb{R}^{m\times n}$. We can evaluate $c_k$ and $J_k$, but use a stochastic estimate $g_k$ of $\nabla f(x_k)$. The analysis assumes that $g_k$ is conditionally unbiased and has bounded variance. It also allows $J_k$ to be rank deficient, a situation that can make a standard SQP linear system singular.

Rank deficiency can arise when some constraints are locally redundant. In that case, insisting that every linearized constraint be solved exactly may not produce a usable search direction. Separating progress toward the constraints from progress on the objective provides a way to continue without assuming full row rank at every iteration.

## How the method works

We split the search direction into a **normal step** $v_k$, which reduces linearized constraint violation, and a **tangential step** $u_k$, which uses the stochastic gradient. The normal step approximately solves

$$
\min_{v\in\mathbb{R}^n}\ \frac12\|c_k+J_kv\|^2
\qquad\text{subject to}\qquad
\|v\|\leq\omega\|J_k^Tc_k\|,
$$

where $\omega>0$ controls its size. We only require sufficient decrease, rather than an exact solution. In particular, the normal step can help even when the linearized constraints cannot be satisfied exactly.

The tangential step approximately minimizes a quadratic model over the null space of $J_k$:

$$
\min_u\ (g_k+H_kv_k)^Tu+\frac12u^TH_ku \quad \mathrm{s.t.} \quad J_ku=0
$$

The symmetric matrix $H_k$ is chosen independently of the current gradient sample and is uniformly positive definite on this null space. If $u_k$ is exact, $J_ku_k=0$, so it does not change the *linearized* constraint residual. For an inexact solve, we stop when the linear-system residuals meet tolerances proportional to a parameter $\beta_k$; the analysis accounts for the remaining error.

We then form two stepsizes:

$$
d_k=v_k+\beta_ku_k,
\qquad
x_{k+1}=x_k+\alpha_kd_k.
$$
The factor $\beta_k$ limits the effect of stochastic gradient noise on the tangential step, while $\alpha_k$ scales the full direction. For the finite-horizon complexity results, the paper takes $\beta_k=\eta/\sqrt K$ and chooses $\alpha_k$ from an interval determined by the problem assumptions. $\beta_k$ must be chosen before an inexact solve because it sets the solver's stopping tolerance.

This is the key reason for using two stepsizes: reducing $\beta_k$ controls the noisy part without shrinking the constraint-oriented normal step by the same amount. It does not make the gradient estimate itself less noisy.

## What the bounds say

A distinction matters when the Jacobian may be rank deficient: $\|c_k\|$ measures **actual constraint violation**, while $\|J_k^Tc_k\|$ measures **stationarity of constraint violation**. The latter can vanish at an infeasible point. For example, with $c(x)=x^2+1$, we have $J(0)^Tc(0)=0$ but $c(0)=1$.

The preprint gives the following iteration orders. Let $R$ be selected uniformly at random from $\{0,\ldots,K-1\}$. The guarantees are in expectation for $x_R$ (equivalently, averages over the first $K$ iterates). They do not say that every run or every iterate meets the tolerance.

| Setting | Constraint-related guarantee | First-order guarantee |
| :-- | :-- | :-- |
| No constraint qualification, with a positive lower bound on the merit parameter | $\mathbb{E}\|J_R^Tc_R\|\leq\varepsilon_c$ in $\mathcal{O}(\varepsilon_c^{-2})$ iterations | Lagrangian-gradient error $\leq\varepsilon_L$ in $\mathcal{O}(\varepsilon_L^{-4})$ iterations |
| Uniform LICQ along the iterates | $\mathbb{E}\|c_R\|\leq\varepsilon_c$ in $\mathcal{O}(\varepsilon_c^{-1})$ iterations | Lagrangian-gradient error $\leq\varepsilon_L$ in $\mathcal{O}(\varepsilon_L^{-4})$ iterations |
| Merit parameter tends to zero, without a constraint qualification | $\mathbb{E}\|J_R^Tc_R\|\leq\varepsilon_c$ in $\mathcal{O}(\varepsilon_c^{-2})$ iterations | No corresponding first-order guarantee from this case |


The first-order measure is the norm of $\nabla f(x_k)+J_k^Ty_k^{\mathrm{true}}$, with a suitable multiplier defined in the paper. Under uniform LICQ, a common positive lower bound on the relevant singular values of $J_k$ lets us control $\|c_k\|$ itself. Without that condition, a small $\|J_k^Tc_k\|$ need not mean near-feasibility. The same orders hold for inexact tangential solves under the paper's solver assumptions and residual tests; see [Sections 3–4 of the preprint](https://arxiv.org/html/2604.14351v1).

The merit parameter is used to analyze progress through $\tau f(x)+\|c(x)\|$; it is not required to compute the two steps above. When its positive lower bound is unavailable, the paper still gives the constraint-stationarity bound in the last row, but does not infer that the iterates become feasible.

## Numerical results

We compared ITSQP with two stochastic SQP baselines, SSQP and a step-lengthening version (SSQPL), on 110 equality-constrained problems from [CUTEst](https://www.numerical.rl.ac.uk/media/people/nick-gould/GoulOrbaToin15_coap.pdf). Four gradient-noise levels give 440 problem-and-noise configurations. The study considers five fixed scaling choices, 15 random seeds, and at most 10,000 iterations per run. The [paper's experiment section](https://arxiv.org/html/2604.14351v1) gives the parameter selection and stopping rules.

For evaluation, an iterate is treated as feasible when its infinity-norm constraint error is at most $10^{-6}$. The reported KKT error uses the true objective gradient available from the test problems, rather than the noisy sample used to compute a step. This makes the plot an assessment of the resulting iterates.

<figure>
  <a href="{{ '/assets/img/posts/complexity%20results.png' | relative_url }}" target="_blank" rel="noopener noreferrer">
    <img
      src="{{ '/assets/img/posts/complexity%20results.png' | relative_url }}"
      class="img-fluid"
      width="2556"
      height="1125"
      loading="lazy"
      alt="Boxplots of final KKT and constraint-violation errors across four noise levels for SSQP, SSQPL, and ITSQP. ITSQP generally has lower constraint violation; KKT performance varies."
    >
  </a>
  <figcaption class="caption">
    Final KKT and constraint-violation errors by noise level, on logarithmic vertical scales. Click for the full-size image.
  </figcaption>
</figure>

In these tests, ITSQP generally achieved lower final constraint violation, especially at higher noise levels. Its KKT error was not uniformly the lowest. Because the figure shows final errors, it does not by itself compare runtime or convergence speed.

For the assumptions, proofs, and complete experiments, see Michael J. O'Neill and Aoji Tang, “[Complexity of an inexact stochastic SQP algorithm for equality constrained optimization](https://arxiv.org/abs/2604.14351),” arXiv:2604.14351 (2026).
