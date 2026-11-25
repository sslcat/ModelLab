# Mathematics of the receipt experiments

I use one receipt as the unit of extraction accuracy and one labeled token as the unit of training loss. Those choices explain how the pilot could reach a very low validation loss and still miss the item field in seven complete records. The examples below use the archived configurations and observations. [Measured results](research.md) · [Machine-readable measurements](../evidence/summary.json)

## From a receipt to a language-model objective

The worker renders a user message with the checkpoint’s chat template. An autoregressive transformer factorizes the probability of the completion tokens given that input. If the target tokens are denoted by $y_t$ and the rendered receipt by $x$, then

$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
$$

Within a self-attention layer, learned projections produce queries, keys and values from hidden states. A causal mask prevents access to future positions. For one attention head with key dimension $d_k$,

$$
Q=HW_Q,\quad K=HW_K,\quad V=HW_V,\qquad
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V .
$$

The output logits generate a categorical distribution over the vocabulary. Cross-entropy is the negative log-probability of the correct next token. If $\mathcal{I}$ contains the shifted, labeled completion positions, the token-averaged loss is

$$
L=-\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log
\frac{\exp z_{i,y_i}}{\sum_{v\in V_{\rm vocab}}\exp z_{i,v}} .
$$

The trainer uses completion-only loss. Prompt and padding positions do not contribute as target tokens. Epoch validation uses teacher forcing, so each next-token prediction is conditioned on the reference prefix. Free generation instead conditions on the model’s preceding outputs. The pilot’s final validation loss was about 0.00023933. A loss of that size corresponds to a token perplexity of about 1.00024. The adapted model still produced only 41/48 correct records on separate test notes. Predicting punctuation, repeated words and familiar targets well can coexist with an incorrect item on a new phrasing. [Training and completion masking](../remote/train.py) · [Pilot observations](../evidence/pilot.json.gz)

## Gradient descent, AdamW and accumulation

Ordinary gradient descent moves trainable parameters in the negative gradient direction. AdamW estimates first and second moments of minibatch gradients and applies decoupled weight decay. With elementwise operations, a step can be written as

$$
g_t=\nabla_\theta L_t,\quad m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2 .
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Here $\eta_t$ is the scheduled learning rate and $\lambda$ is the configured decay coefficient. This is the optimizer family used by the worker through Torch AdamW. Only the adapter parameters participate in the update. The runs started at 0.0002. I retain the measured learning-rate series in each session file. [Worker configuration](../remote/train.py) · [AdamW](https://arxiv.org/abs/1711.05101)

Gradient accumulation combines gradients from successive microbatches before an optimizer update. For a token-averaged objective, if microbatch $j$ has $n_j$ labeled positions and mean loss $L_j$, the combined gradient is

$$
g=\frac{\sum_{j=1}^a n_j\nabla L_j}{\sum_{j=1}^a n_j}.
$$

For equal token counts this is the arithmetic mean of the microbatch gradients. Example counts alone do not specify the token weighting. With one GPU, microbatch size one and accumulation eight, a full update covers eight examples. The rank and size studies used 384 training examples for one epoch, giving 48 optimizer updates. The pilot used 204 and logged 26 updates, including the final partial group. Peak CUDA allocation, driver-sampled device memory, training tokens/s and generation tokens/s measure different parts of the workload. [Training histories and dataset definitions](../evidence/README.md)

## Low-rank adaptation and trainable counts

A dense matrix of rank at most $r$ admits a factorization through an $r$-dimensional space. For example, truncating an SVD gives the best approximation of rank $r$ under squared Frobenius error. LoRA learns two factors by task loss. It does not compute the pretrained matrix’s SVD. In the linear-layer convention below, $W_0$ has shape $d_{\rm out}\times d_{\rm in}$. The deployed update and its rank bound are

$$
W=W_0+\frac{\alpha}{r}BA,\quad
A\in\mathbb R^{r\times d_{\rm in}},\quad
B\in\mathbb R^{d_{\rm out}\times r},\quad
\mathrm{rank}(BA)\le r .
$$

The update adds $r(d_{\rm in}+d_{\rm out})$ trainable values per targeted matrix. In the rank sweep I used ranks 4, 8, 16 and 32 with alpha 8, 16, 32 and 64, keeping the multiplier at two. The dropout probability on the adapter path was 0.05 during training. During evaluation the worker switches the model to evaluation mode.

The Qwen backbones use grouped-query attention. In each of $L$ layers the query projection maps hidden width $h$ to $h$. The value projection maps it to $h n_{\rm kv}/n_{\rm heads}$. Summing the four LoRA factors per layer gives

$$
N_{\rm adapter}=rL\left(3h+h\frac{n_{\rm kv}}{n_{\rm heads}}\right).
$$

For Qwen2.5-7B, width 3,584, 28 layers, 28 query heads and four key/value heads give a value width of 512. Rank four therefore yields

$$
4\cdot28\cdot(3\cdot3584+512)=1{,}261{,}568 .
$$

At rank 32 the count is 10,092,544. The two measured adapter files were 5,060,984 and 40,385,208 bytes. File length includes tensor headers. Rank four used about 0.0166% as many trained values as the 7,615,616,512-parameter backbone. That ratio says nothing by itself about activation memory, latency or task accuracy. [Model geometries](../research/size_models.json) · [Serialized tensor checks](../research/size_diagnostics.py) · [LoRA](https://arxiv.org/abs/2106.09685)

## Four-bit quantization and QLoRA

The frozen backbone’s eligible weight blocks are represented with 4-bit NF4 codes. At a conceptual level the decoder looks up a codebook value $c_k$ and applies a block scale $s_b$.

$$
\tilde w_i=s_b c_{k_i},\qquad k_i\in\lbrace 0,\ldots,15\rbrace.
$$

Double quantization compresses scale information as well. The worker requests NF4, double quantization and BF16 compute. The adapter learns against the dequantized approximation during layer computation. The frozen matrix and its codes receive no optimizer updates. A useful expression for the adapted output during inference is

$$
z=\tilde W_0 x+\frac{\alpha}{r}B(Ax).
$$

This representation reduces weight storage on the remote GPU. A four-bit code requires half a byte, but block metadata, non-quantized layers, optimizer state, activations and CUDA workspaces also consume memory. The final size study logged a 7B training allocator peak around 8,426 MiB while all three adapters had the same 5,060,984-byte serialized size. Adapter file size cannot predict total GPU memory. Both base and adapted controls used the same quantized backbone, so this experiment did not isolate quantization error. [Implementation](../remote/train.py) · [QLoRA](https://arxiv.org/abs/2305.14314)

## Constrained generation and extraction scores

Let $S_t$ be the tokens permitted by the JSON grammar after the current prefix. In the controlled runs, greedy decoding selects

$$
y_t=\mathop{\rm argmax}_{v\in S_t} z_{t,v}.
$$

The grammar requires the keys and types for item, quantity and site. It accepts arbitrary strings and integer quantities. It has no access to the target values. The generation ceiling was 96 new tokens. Invalid or truncated responses remain in the denominator. [Grammar implementation](../research/size_worker/controlled.py)

The strict parser rejects duplicate keys for schema scoring and checks exact Python value types. Let $s_i$ be a schema pass and $f_{ij}$ indicate that field $j$ matches the reference with the correct type. Then

$$
a_{\rm record}=\frac{1}{n}\sum_i s_i\prod_{j=1}^3 f_{ij},\qquad
a_j=\frac{1}{n}\sum_i f_{ij}.
$$

A wrong receiving site makes a complete record incorrect even when both other fields match. The primary 1.5B specialist passed schema on 256/256 locked notes and got 155 items, 156 quantities and 161 sites right. Only 130 notes had all three right. Removing a single enclosing Markdown fence was a separate diagnostic in the pilot. It preserves the extraction values and changes which outputs can be parsed. [Scoring and parser](../remote/receipt_scores.py) · [Field counts](../evidence/scores.csv)

## Paired comparisons and confidence intervals

Each candidate and reference see the same notes. Count $g$ when only the candidate is right and $\ell$ when only the reference is right. Ties contribute to the denominator of the rate difference and drop out of the exact McNemar test.

$$
\hat\Delta=\frac{g-\ell}{n},\qquad
p=\min\left(1,2\sum_{j=0}^{\min(g,\ell)}
{g+\ell\choose j}2^{-(g+\ell)}\right).
$$

The test conditions on the number of discordances. Under its null, either model wins a discordant comparison with probability one half. For the primary 1.5B versus 7B comparison, 58 gains and 38 losses among 256 receipts yield a 7.8125-point gain and a two-sided p-value of 0.05191.

For the registered difference interval, the code inverts exact binomial tails separately for the gain and loss probabilities. Each marginal interval has 97.5% nominal coverage, with 0.0125 in each tail. Bonferroni gives at least 95% joint coverage under the independent-note sampling model. If the two intervals are $[L_g,U_g]$ and $[L_\ell,U_\ell]$, the reported difference interval is

$$
[L_g-U_\ell,\ U_g-L_\ell].
$$

The observed primary interval was −3.50 to +18.88 percentage points. It admits both signs. The p-value and conservative interval are different inferential constructions. Correlation among authored templates weakens population interpretations. Equality of two observed accuracies, or failure to reject a difference, cannot establish equivalence. Equivalence would require a defensible margin and a design with sufficient precision. [Exact tail inversion and matching](../research/study_results.py)

## Training seeds and resource trade-offs

For $S$ seeds, I use the arithmetic mean and sample standard deviation of their accuracies. Three seeds on one set of notes yield three trained models and one shared benchmark.

$$
\bar a=\frac{1}{S}\sum_{s=1}^S a_s,\qquad
s_a=\sqrt{\frac{\sum_s(a_s-\bar a)^2}{S-1}} .
$$

The 1.5B adapters got 130, 124 and 123 of 256 records right. This gives a mean of 49.09% and sample SD of 1.48 percentage points. The primary seed was fixed at 42 before evaluation. Changing the reported seed to whichever won would change the estimand. The three observations provide a coarse estimate of seed variability. [Per-seed calculations](../research/public_results.py)

For a frontier, let $a$ denote accuracy, $t$ generation latency and $c$ the full session charge. Point $u$ dominates $v$ when all these relations hold and at least one is strict.

$$
a_u\ge a_v,\qquad t_u\le t_v,\qquad c_u\le c_v .
$$

The frontier contains all undominated eligible points, retaining exact ties. A missing charge excludes a point from a cost comparison. Tuned accuracy is the seed mean, and tuned latency is the mean of per-seed medians of synchronized, batch-one generation time. On the observed joint frontier, tuned 3B traded lower accuracy for shorter latency, 3B base shared its session charge with higher accuracy and longer latency, and tuned 7B supplied the highest mean accuracy. These points describe the four measured hosts and receipt mix. Whole-session cost includes setup, training and evaluation and cannot identify marginal production inference cost. [Frontier implementation](../research/size_results.py)
