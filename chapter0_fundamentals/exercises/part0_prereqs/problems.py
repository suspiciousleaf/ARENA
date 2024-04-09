# %%
import os
import sys
import math
import numpy as np
import einops
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part0_prereqs"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part0_prereqs.utils import display_array_as_img
import part0_prereqs.tests as tests

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
# %%
import utils

for img in arr:
    utils.display_array_as_img(img)

# %%
# Exercise 1: Display concatenated images.

arr1 = einops.rearrange(arr, "i c h w -> c h (i w)")
utils.display_array_as_img(arr1)

# %%
# Exercise 2: Vertically concatenate two 0 images.
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
utils.display_array_as_img(arr2)

# %%
arr3 = einops.repeat(arr[0:2], "i c h w -> c (i h) (rep_w w)", rep_w=2)
utils.display_array_as_img(arr3)

# %%

arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
utils.display_array_as_img(arr4)

# %%
# Exercise 5: Splitting color channels.
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
utils.display_array_as_img(arr5)

# %%
# Exercise 6: Wrap images across two rows.
arr6 = einops.rearrange(arr, "(i1 i2) c h w -> c (i1 h) (i2 w)", i1=2)
utils.display_array_as_img(arr6)

# %%
# Exercise 7: Max of color channels
arr7 = einops.reduce(arr, "i c h w -> h (i w)", "max")
utils.display_array_as_img(arr7)

# %%
# Exercise 8: Stack images and take minimum.
arr8 = einops.reduce(arr, "i c h w -> h w", "min")
utils.display_array_as_img(arr8)

# %%
# Exercise 9: Transpose
arr9 = einops.rearrange(arr[1], "c h w -> c w h")
utils.display_array_as_img(arr9)

# %%
# Exercise 10: Splitting i, concatenating v and h, scaling down by 2

arr10 = einops.reduce(
    arr.astype(float), "(i1 i2) c (h 2) (w 2) -> c (i1 h) (i2 w)", "mean", i1=2
)
utils.display_array_as_img(arr10)


# %%
# Einops exercises - helper functions
def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


def assert_all_close(
    actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001
) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


# %%
# Exercise A.1 - rearrange
def rearrange_1() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    tmp = t.arange(3, 9)
    result = einops.rearrange(tmp, "(b1 b2) -> b1 b2", b1=3)
    return result


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


# %%
# Exercise A.2 - arange and rearrange
def rearrange_2() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    tmp = t.arange(1, 7)
    result = einops.rearrange(tmp, "(b1 b2) -> b1 b2", b1=2)
    return result


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


# %%
# Exercise A.3 - rearrange
def rearrange_3() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    """
    tmp = t.arange(1, 7)
    result = einops.rearrange(tmp, "b -> 1 b 1")
    return result


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))


# %%
# Exercise B.1 - temperature average
def temperatures_average(temps: t.Tensor) -> t.Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(n 7) -> n", "mean")


temps = t.Tensor(
    [71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83]
)
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)


# %%
# Exercise B.2 - temperature difference
def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    means = einops.repeat(einops.reduce(temps, "(n 7) -> n", "mean"), "n -> (n 7)")
    return temps - means


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)


# %%
# Exercise B.3 - temperature normalized
def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """
    means = einops.reduce(temps, "(t 7) -> t", "mean")
    diffs = temps - einops.repeat(means, "t -> (t 7)")
    stds = einops.reduce(diffs, "(d 7) -> d", t.std)
    return diffs / einops.repeat(stds, "s -> (s 7)")


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)


# %%
# Exercise C - identity matrix
def identity_matrix(n: int) -> t.Tensor:
    """Return the identity matrix of size nxn.

    Don't use torch.eye or similar.

    Hint: you can do it with arange, rearrange, and ==.
    Bonus: find a different way to do it.
    """
    assert n >= 0
    seq = t.arange(n)
    seq_1 = einops.repeat(seq, "i -> n i", n=n)
    seq_2 = einops.repeat(seq, "i -> i n", n=n)
    result = seq_1 == seq_2
    return result


assert_all_equal(identity_matrix(3), t.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
assert_all_equal(identity_matrix(0), t.zeros((0, 0)))


# %%
# Exercise D - sample distribution
def sample_distribution(probs: t.Tensor, n: int) -> t.Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use torch.rand and torch.cumsum to do this without any explicit loops.

    Note: if you think your solution is correct but the test is failing, try increasing the value of n.
    """
    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()
    rs = t.rand(n)
    cs = t.cumsum(probs, dim=0)
    is_bigger = rs[:, None] > cs
    return einops.reduce(is_bigger, "r i -> r", "sum")


n = 10000000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs, rtol=0.001, atol=0.001)


# %%
# Exercise E - classifier accuracy
def classifier_accuracy(scores: t.Tensor, true_classes: t.Tensor) -> t.Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use torch.argmax.

    Find index of max value of each sub list( [0.75, 0.5, 0.25] max index is 0), see if it matches the index in true_classes, return the fraction of correct / total
    """
    assert true_classes.max() < scores.shape[1]
    ans = (scores.argmax(dim=1) == true_classes).float().mean()

    return ans


scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected


# %%
# Exercise F.1 - total price indexing
def total_price_indexing(prices: t.Tensor, items: t.Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k). Each value is the index of a single item purchased, return total price of order.

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    assert items.max() < prices.shape[0]

    # return sum([prices[item] for item in items])[0]
    ans = prices[items].sum().item()

    return ans


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0

# %%
# Exercise F.2 - gather 2D


def gather_2d(matrix: t.Tensor, indexes: t.Tensor) -> t.Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    For diagram see (note diagram uses 1 index, code uses 0 index) https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    """

    assert matrix.ndim == indexes.ndim
    assert indexes.shape[0] <= matrix.shape[0]
    out = matrix.gather(1, indexes)
    assert out.shape == indexes.shape
    return out


matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)
indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)

# %%
# Exercise F.3 - total price gather


def total_price_gather(prices: t.Tensor, items: t.Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    ans = prices.gather(0, items).sum().item()
    return ans


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0

# %%
# Exercise G - indexing


def integer_array_indexing(matrix: t.Tensor, coords: t.Tensor) -> t.Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )

    For integer array indexing, to access three values at [0, 0], [1, 2], [3, 4]:
    matrix[(0, 1, 3), (0, 2, 4)]. Indicies will be looped through to return [0, 0], then [1, 2] etc
    """
    ans = matrix[tuple(coords.T)]
    return ans


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))
mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))

# %%
# Exercise H.1 - batched logsumexp


def batched_logsumexp(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

    (C) Find max value for each row, (D) rearrange into 2D matrix
    ([[max_row_1], [max_row_2]] instead of [max_row_1, max_row_2])
    (E) subtract row max from each value in row
    (exps) find exponent of each value
    (F) sum along each row of exponents and take the log
    (ans) add this to the initial row sum, (C)

    """

    C = matrix.max(dim=-1).values
    D = einops.rearrange(C, "n -> n 1")
    E = matrix - D
    exps = t.exp(E)
    F = t.log(t.sum(exps, dim=-1))
    ans = C + F
    return ans


matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)
matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)


# %%
# Exercise H.2 - batched softmax
def batched_softmax(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.

    (exp) Calculate exponent for each value in tensor
    (ans) Divive each exp value by the sum of the row of exp values, to normalize (get its proportion of the total sum, so total sum is mapped to 1 and each value is its relative proportion of that)
    """
    exp = matrix.exp()
    ans = exp / exp.sum(dim=-1, keepdim=True)
    return ans


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))
matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))

# %%
# Exercise H.3 - batched logsoftmax


def batched_logsoftmax(matrix: t.Tensor) -> t.Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    (A) Find max per row
    (B) subtract row max from actual values
    (ans) Compute softmax for each row, then take log and return
    """
    A = matrix.max(dim=-1).values
    B = matrix - A
    ans = batched_softmax(B).log()

    return ans


matrix = t.arange(1, 6).view((1, 5)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 6).view((1, 5)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-4.4519, -3.4519, -2.4519, -1.4519, -0.4519]])
assert_all_close(actual, expected)

# %%
# Exercise H.4 - batched cross entropy loss


def batched_cross_entropy_loss(logits: t.Tensor, true_labels: t.Tensor) -> t.Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    #! Answer taken from solutions on https://arena3-chapter0-fundamentals.streamlit.app/[0.0]_Prerequisites, does not pass tests
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]

    logprobs = batched_logsoftmax(logits)
    indices = einops.rearrange(true_labels, "n -> n 1")
    pred_at_index = logprobs.gather(1, indices)
    return -einops.rearrange(pred_at_index, "n 1 -> n")


logits = t.tensor(
    [[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]]
)
true_labels = t.tensor([2, 0, 0])
expected = t.tensor([0.0, math.log(3), float("inf")])
actual = batched_cross_entropy_loss(logits, true_labels)
assert_all_close(actual, expected)

# %%
# Exercise I.1 - collect rows


def collect_rows(matrix: t.Tensor, row_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    assert row_indexes.max() < matrix.shape[0]

    ans = matrix[row_indexes]
    return ans


matrix = t.arange(15).view((5, 3))
row_indexes = t.tensor([0, 2, 1, 0])
actual = collect_rows(matrix, row_indexes)
expected = t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
assert_all_equal(actual, expected)

# %%
# Exercise I.2 - collect columns


def collect_columns(matrix: t.Tensor, column_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    Transpose matrix, select by row as above, transpose result to get back into original format
    Alternatively, [:, column_indexes] selects all rows, but only the columns specified
    """
    assert column_indexes.max() < matrix.shape[1]
    ans = matrix.T[column_indexes].T
    ans = matrix[:, column_indexes]
    return ans


matrix = t.arange(15).view((5, 3))
column_indexes = t.tensor([0, 2, 1, 0])
actual = collect_columns(matrix, column_indexes)
expected = t.tensor(
    [[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]]
)
assert_all_equal(actual, expected)


# %%
# Einsum
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    # ident = t.eye(mat.shape[0], mat.shape[1])
    # return einops.einsum(mat * ident.numpy(), "r c -> ")
    return einops.einsum(mat, "i i -> ")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec[:, None], "a b, c d -> a d")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    pass


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    pass


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    pass


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)
# %%
