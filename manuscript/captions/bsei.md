BSEI. The concatenated skip
tensor is projected and refined with a depthwise-separable convolution. A
depthwise convolution then generates a response independently for every output
channel. The sigmoid response multiplicatively enhances the refined feature in
a residual form. All 3 x 3 convolutions use padding one, preserving spatial
resolution.
