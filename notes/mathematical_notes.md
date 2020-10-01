## Abstracting

Many social circles develop natural hierarchies through competition, cooperation, award sharing or other metrics of prestige. Recently developed methods have shown the ability to temporally reconstruct these hierarchies based on the assumption on the mechanisms which determine how individuals within a given social network form their connection. We extend this method to take in various possible covariates and generate features which can be used to recreate hierarchical structure. We show that these aforementioned models of emergent hierarchy can be combined with statistical models in order to generate rankings among individuals. We use a simple liner model involving several hand-picked covariates of actor-director connection formation and use them to simulate an emerging hierarchy of actors and directors in the movie industr y. This shows....

## Generating actor-director social networks

## Dynamic model of hierarchy in Hollywood

The idea of endorsement is modeled by matrices $A(t)$ which are indexed by each step $t$. Endorsements evolve in time according to:

$$
\begin{equation}
A(t + 1) = \lambda A(t) + (1 - \lambda) \Delta(t)
\end{equation}
$$

This means that the endorsement history in each time step is a weighted average of new endorsements and past endorsement history.

The rank vector is given by:
$$
\begin{equation}
\gamma_i = \frac{1}{n} \sum_{i} p_{ij} 
\end{equation}
$$

We differ from Kawakatsu et al's formula significantly in that we model the utility associated with individuals $i$ and $j$ endorsing one another according to several handpicked covariates $x_1, \ldots, x_n$, so that 

$$
\begin{equation}
u_{ij} = \beta_1 x_1 + \cdots + \beta_n x_n
\end{equation}
$$
 
<!--- We hope that by using various covariates we can avoid the issue of directly proposing a mechanism driving heirarchy formation. -->

## Inference

Our data inputs here are our actor-director social network $\Delta(t)$ and the various hand-selected covariates $x_k$.

Our goal for inference is to find the parameter vector $\vec{\beta}$ and memory parameter $\lambda$ best describing this dynamic Formally, we can write this as: 

$$
\begin{equation}
P( \{\Delta(t) \} \mid A(0), \lambda, \beta ) = \prod P(\Delta(t) \mid A(t), \beta) 
\end{equation}
$$

We can write the final likelihood as
$$
\begin{equation}
L(\lambda, \vec{\beta}, \mid \{ \Delta(t)\}) = \sum_{i,j,t} k_{ij}(t) \log \gamma_{ij}(t) + C
\end{equation}
$$

Note: This likelihood is convex in $\vec{\beta}$ as it is a linear model. Therefore, we can find optimize first with respect to $\vec{\beta}$ with standard methods and then using $\lambda$ with gradient ascent. 
