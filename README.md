In this repo I explore the concept of n-way-attention, in particular 3-way-attention.

In the classical attention algorithm, the attention layer sums over every pair of tokens. We can also view it as, for every given token, we sum a term that depends on each of the other tokens (previous tokens in the autoregressive case) and the token itself. The idea of 3-way-attention is to sum over every trio of tokens. Or, for each given token to sum over every pair of tokens that has come before.
There have been many variations on the original attention algorithm, many of them with the intent of making it asympotitaclly more efficient. I have not seen this idea, which makes attention even slower, implemented before.

The first interesting thing is that, at least to me, there no clear or natural way to do it. So I have come up with two implementations, but I am sure there are many more.

In the simpler one, that I call "Trittention" (see trittention.py), the attention score is the result of the generalized dot product of the two keys and the query. And the value is just the sum of the two 'value' vectors. The formula is roughly:

$$
\begin{align*}
k_i &= K_1 \cdot \text{token}_i \\
k_j &= K_2 \cdot \text{token}_j \\
q_h &= Q \cdot \text{token}_h \\
v_i &= V_1 \cdot \text{token}_i \\
v_j &= V_2 \cdot \text{token}_j
\end{align*}
$$

The new token is computed as:

$$
{newtoken}_h = \sum \left( \sum \left( \text{softmax}\left( q_h \cdot k_i \cdot k_j \right) \right) \cdot \left( v_i + v_j \right) \right)
$$

Where the original attention would have been something like:

$$
\begin{align*}
k &= K \cdot \text{token}_i \\
q &= Q \cdot \text{token}_h \\
v &= V \cdot \text{token}_i
\end{align*}
$$

The new token is computed as:

$$
{newtoken}_h = \sum \left( \text{softmax}\left( q_h \cdot k_i) \right) \cdot \left( v_i \right) \right)
$$

(The softmax is computed over all of the "scores" before the sumation)

The other way I have implemented it (see trittention_cube.py) can be understood with two tensors of size (d_model, d_model, d_model).

The score:

$$
{score} = K_{i,j,h} \cdot q_h \cdot k_i \cdot k_j
$$

The value:

$$
{value} = V_{i,j,k} \cdot v_i \cdot v_j
$$

And new_token:

$$
{newtoken}_h = \sum \left( \sum \left( \text{softmax}\left( score \right) \right) \cdot \left( value \right) \right)
$$


In the actual implementation the tensors are of size (d_head, d_head, d_head) and the residual stream is multiplied before by matrices of size (d_model, d_head), like in standard attention.


### Results

## One-layer Induction Head



## Toy problems

## Equivalence

## Language Model

