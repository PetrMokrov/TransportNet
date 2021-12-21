# [Stable Dynamic & Beckmann models](https://github.com/MeruzaKub/TransportNet/tree/master/Stable%20Dynamic%20%26%20Beckman)

The project contains implementations of several primal-dual subgradient methods for searching traffic equilibria in the Stable Dynamics model and the Beckmann model. 
Results of experiments on the [Anaheim transportation network](https://github.com/bstabler/TransportationNetworks) are included.

The following methods are implemented:
1.	Universal gradient method [[ref](http://www.optimization-online.org/DB_FILE/2013/04/3833.pdf)]
2.	Universal method of similar triangles [[arXiv:1701.02473](https://arxiv.org/ftp/arxiv/papers/1701/1701.02473.pdf)]
3.  Method of Weighted Dual Averages [[ref](https://ium.mccme.ru/postscript/s12/GS-Nesterov%20Primal-dual.pdf)]
4.	Subgradient method with adaptive step size [[arXiv:1604.08183](https://arxiv.org/ftp/arxiv/papers/1604/1604.08183.pdf)].

Convergence rates of UMST, UGM, composite and non-composite WDA-methods for the Stable Dynamics model:

<img src="https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/pics/sd_convergence_rel_eps.jpg" width="500">

Convergence rates of UMST, UGM, composite and non-composite WDA-methods, and the Frank–Wolfe method for the Beckmann model:

<img src="https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/pics/beckmann_convergence_rel_eps.jpg" width="500">

## Usage of `T-SWSF` for stable dynamic problem
One can substitute the `Dijkstra` algorithm when performing flow reconstruction phase (see `Alogorithm 1` of the original [article](https://arxiv.org/pdf/2008.02418.pdf)) with the `T-SWSF` algorithm [[ref](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.5911&rep=rep1&type=pdf)] aimed at solving the Dynamic Single-Source Shortest-Path problem. The general idea behind this problem is to utilize the knowledge of shortest paths and shortest distances computed on the previous step for computing the shortest paths and shortest distances on the current step (after graph edge's weights perturbation due to gradient step of a stable dynamic problem solver). 

### `T-SWSF` analysis

Our experimental study shows, that for arbitrary graph perturbations the `T-SWSF` performs worse than `Dijkstra` (for fair comparison we implement both methods in python, our implementations are in the following [file](./Stable%20Dynamic%20%26%20Beckman/t_swsf.py)). However, if the ratio of the number of perturbed edges to the number of all edges is sufficiently small, the `T-SWSF` works faster than recomputation of shortest paths and distances from scratch using `Dijkstra` algorithm. We use the `ratio` constant equals to $r^* = \frac{1}{20}$ but probably it is not the optimal one (and depends on implementation).  We refer to the graph perturbation situation with $r \leq r^*$ as *sparse graph perturbation*.

### `T-SWSF` application

Our experiment study shows that in the stable dynamic model when using `Universal Method of Similar Triangles`(`ustm`) and `Universal gradient method`(`ugm`) solvers (see sections $3.3$ and $3.4$ of the original [article](https://arxiv.org/pdf/2008.02418.pdf) ) the *sparse graph perturbation* situations occurs frequently (every third or even every second graph update is actually *sparse*) (see the [notebook](./Stable%20Dynamic%20%26%20Beckman/Anaheim_Stable_Dynamics_Experiments.ipynb) for reference), and therefore the utilization of `T-SWSF` in such cases seems profitable. 

In order to validate the profitability of the `T-SWSF` when solving stable dynamic model using `ustm` and `ugm` we compare the time needed to achieve fixed duality gaps $\varepsilon$-s (following the original research by `Kubentayeva et. al.`). Our experiments are available in the [notebook](./Stable%20Dynamic%20%26%20Beckman/Anaheim_SD_T_SWSF.ipynb). The final chart is the following:

<img src="./Stable%20Dynamic%20%26%20Beckman/pics/t_swsf_profit.png" width="500">

The `_dijkstra` suffix means, that we solve the SSSP problems by recomputing all paths from scratch using `Dijkstra` algorithm, the `_t_swsf` suffix means, that we apply `T-SWSF` for arbitrary graph perturbations (even for $r > r^*$). The `_tradeoff` means, that we use `Dijkstra` if $r > r^*$ and `T-SWSF` if $r \leq r^*$. Our numerical experiment shows that indeed the `T-SWSF` could improve the time complexity in the considered cases. 

### Further directions

The `T-SWSF` is not the only choice for solving Dynamic Single-Source Shortest-Path problem (and not the most efficient one, especially given our implementation). The potential algorithms which could solve the problem (probably, more efficient) are as follows:

* [[https://www.sciencedirect.com/science/article/pii/S1319157817303828](https://www.sciencedirect.com/science/article/pii/S1319157817303828)]

* [[https://arxiv.org/pdf/1504.07091.pdf](https://arxiv.org/pdf/1504.07091.pdf)]

* [[https://arxiv.org/pdf/1409.6241.pdf](https://arxiv.org/pdf/1409.6241.pdf)]

(It is not the full list, just what we have found)

## Installing graph-tool
Native installation of [graph-tool](https://graph-tool.skewed.de/) on Windows isn't supported. But if you have Docker installed, you can easily download the following container image with all the packages required to run the project:
https://hub.docker.com/r/ziggerzz/graph-tool-extra 

## How to Cite
1. Kubentayeva, M.; Gasnikov, A. Finding Equilibria in the Traffic Assignment Problem with Primal-Dual Gradient Methods for Stable Dynamics Model and Beckmann Model. Mathematics 2021, 9, 1217. https://doi.org/10.3390/math9111217
2. The source code: Kubentayeva M. TransportNet. https://github.com/MeruzaKub/TransportNet. Accessed Month, Day, Year.

## More Resources
More information about the models can be found in [[Nesterov-de Palma](https://link.springer.com/article/10.1023/A:1025350419398)] and [[Beckmann](https://cowles.yale.edu/sites/default/files/files/pub/misc/specpub-beckmann-mcguire-winsten.pdf)].

# [Stochastic Nash-Wardrop Equilibria in the Beckmann model](https://github.com/MeruzaKub/TransportNet/tree/master/Stochastic%20Nash-Wardrop%20equilibrium)
Agents’ behavior is not completely rational, what is described by the introduction of Markov logit dynamics: any driver selects a route randomly according to the Gibbs’ distribution taking into account current time costs on the edges of the graph.
<img src="https://render.githubusercontent.com/render/math?math=\gamma > 0"> is a stochasticity parameter (when <img src="https://render.githubusercontent.com/render/math?math=\gamma \rightarrow 0"> the model boils down to the ordinary Beckmann model). The figure below shows convergence of flows in stochastic equilibrium to equilibrium flows in non-stochastic case as  <img src="https://render.githubusercontent.com/render/math?math=\gamma"> tends to zero.

<img src="https://github.com/MeruzaKub/TransportNet/blob/master/Stochastic%20Nash-Wardrop%20equilibrium/pics/anaheim_error_vs_gamma_eps_1e-3.png" width="500">

## How to Cite
1. [Article](http://crm.ics.org.ru/uploads/crmissues/crm_2018_3/2018_01_07.pdf): Gasnikov A.V., Kubentayeva M.B. Searching stochastic equilibria in transport networks by universal primal-dual gradient method // Computer Research and Modeling, 2018, vol. 10, no. 3, pp. 335-345. DOI: 10.20537/2076-7633-2018-10-3-335-345.
2. The source code: Kubentayeva M. TransportNet. https://github.com/MeruzaKub/TransportNet. Accessed Month, Day, Year.

<!--- Convergence process on 10 000 iterations for Stable Dynamic model:--->
<!---![](methods_stable_dynamic.png)--->

<!---Convergence process on 8000 iterations for Beckmann model (+ Frank-Wolfe algorithm):--->
<!---![](methods_beckmann.png)--->

<!--[Anaheim_Experiments.ipynb](https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/Anaheim_Experiments.ipynb) contains code of experiments on comparison of the above methods and Frank-Wolfe algorithm (only for the Beckmann model).-->
