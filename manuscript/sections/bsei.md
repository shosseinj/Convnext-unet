# BSEI method text ready for Word review

BSEI is applied after the
decoder and encoder features have been concatenated at a skip level. Let
X_l in R^(C_in x H_l x W_l) denote this concatenated tensor. The module first
projects the channels and applies channel-first layer normalization and GELU:

Z_l = GELU(LN(Conv_1x1(X_l))).

After spatial dropout, a depthwise-separable 3 x 3 convolution followed by
layer normalization and GELU produces Y_l in R^(C_out x H_l x W_l):

Y_l = GELU(LN(SepConv_3x3(Dropout(Z_l)))).

A depthwise 3 x 3 convolution with one group per output channel produces a
channel-wise response. The absolute response is mapped to [0, 1] with a
sigmoid and modulates Y_l through a residual multiplicative update:

E_l = sigmoid(abs(DWConv_3x3(Y_l))),
O_l = Y_l + E_l elementwise-multiplied-by Y_l.

Both E_l and O_l have C_out channels. Padding of one pixel in both 3 x 3
operations preserves H_l and W_l. Thus, the audited implementation does not
use a single spatial response broadcast over channels.
