# Training objective text ready for Word review

Let z denote the predicted logit, p = sigmoid(z) the foreground probability,
and y the binary reference mask. The shared objective for all ablation variants
is

L_seg = 0.30 L_Dice + 0.30 L_BCE + 0.40 L_boundary.

The soft Dice term uses a smoothing constant of one:

L_Dice = 1 - mean_b [(2 sum_i p_bi y_bi + 1) /
                     (sum_i p_bi + sum_i y_bi + 1)].

For the unweighted BCE term, the target is smoothed with epsilon = 0.02:

y_s = (1 - epsilon)y + 0.5 epsilon.

Boundary weights are computed from a 3 x 3 morphological band:

w = 1 + 5 clamp(dilate_3x3(y) - erode_3x3(y), 0, 1).

The boundary term is the sum of weighted BCE and weighted IoU loss:

L_boundary = [sum_i w_i BCE(z_i, y_i) / sum_i w_i]
             + mean_b [1 - (sum_i w_i p_i y_i + 1e-6) /
                            (sum_i w_i(y_i + p_i - y_i p_i) + 1e-6)].

When deep supervision is enabled, the main prediction and up to three auxiliary
predictions use weights 1.00, 0.10, 0.05, and 0.02, respectively. Variants with
fewer outputs consume only the corresponding prefix of this fixed weight list.
