# N-Way-Attention

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
{score} = K_{i,j,h} \cdot token_h \cdot token_i \cdot token_j
$$

The value:

$$
{value} = V_{i,j,k} \cdot token_i \cdot token_j
$$

And new_token:

$$
{newtoken}_h = \sum \left( \sum \left( \text{softmax}\left( score \right) \right) \cdot \left( value \right) \right)
$$


In the actual implementation the tensors are of size (d_head, d_head, d_head) and the residual stream is multiplied before by matrices of size (d_model, d_head), like in standard attention.


## Some Results

### One-layer Induction Head

Induction heads are attention heads that solve the problem "[A][B] ... [A]->[B]", if a token [A] was followed by another [B] previously in the input, current token [A] may be followed by [B] again. They are believed to be key for in-context learning. In the first article desccribing them the circuit involved consisted of two layers. It seemed to me that given that 3-way-attention attends to two key value, they could perform this in one layer. So I trained a one layer "Triformer" and analyze it in "induction_head.ipynb", and it does appear to have induction heads performing this operation in one layer.

This is not super extraordinary, as a normal attention head could also perform induction in one layer if the key and value tokens are shifted by one.

### Toy problems

When comparing attention only transformers (no mlp layer) one domain I have seen where 3-way-attention performs much better is "hard" problems, or problems where the solution is not O(n). For example a string of simple arithmetic operations "3+(7-9)*2+19=" is solvable in O(n), and both perform similarly. However finding the longest increasing subsequence is O(n*log(n)) and there is a difference in performance.

Here is a table comparing them by the maximum accuracy achieved.

| LIS Input Length | Attention 10 Layers  | Attention 7 Layers  | Trittention 4 Layers | TrittentionCube 4 Layers | TrittentionCube 3 Layers |
|------------------|----------------------|---------------------|----------------------|--------------------------|--------------------------|
| 14               | 0.993                | 0.974               | 0.992                | 0.996                    | 0.988                    |
| 21               | 0.989                | 0.945               | 0.973                | 0.990                    | 0.964                    |
| 27               | 0.926                | 0.880               | 0.895                | 0.969                    | 0.918                    |
| 37               | 0.910                | 0.742               | 0.713                | 0.934                    | 0.870                    |

The input length is the length of characters the model receives as input, and is always the same for one run. The longest increasing subsequence is distributed between 1 and the input length.
In this case all models had d_model = 192, d_head =32 and n_head=6. So the trittention models have more parameters per layer.

The difference in performance is reduced (not completely) if a mlp layer is added.


### Equivalence

One natural question is to wonder if whatever trittention is doing, normal attention can do it in 2 layers (like induction) or more. If the difference is not too big, then trittention is not worth it.
In "compare.py" I initialize a layer at random and train another one to try to match random inputs. I do not have a comprehensive table, but a quick result is that, for same number of **total** parameters, one layer trittentionCube learns two layers of attention better than viceversa.

### Language Model

The complexity of trittention is O(n^3), so its unusable for any kind of text sequence. I have implemented a local attention version in local_trittention.py. And in mixed_attention.py an attention layer that has a number of local trittention heads and normal attention heads.
I am training some models with mixed attention, but I have limited compute and they are not very impressive so far.

## Acknowlegments
I have copied from the following repos:

- [LucidRains Local Attention](https://github.com/lucidrains/local-attention/tree/master)
- [Google Gemma PyTorch](https://github.com/google/gemma_pytorch/tree/main)
- [Callum McDougall's ARENA 2.0](https://github.com/callummcdougall/ARENA_2.0)

The concept I had not see it before, but that may be entirely my fault.


