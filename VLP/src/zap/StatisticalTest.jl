# phuonglh@gmail.com
# Test of statistical significance

using HypothesisTests

# GRU scores
xs = [0.550967, 0.564895, 0.574058, 0.562574, 0.550438]
μ0 = mean(xs)

# GRU+ scores
ys = [0.578619, 0.582977, 0.573570, 0.573814, 0.57361]

# Perform a paired sample t-test of the null hypothesis that the differences between pairs of values in vectors x and y 
# come from a distribution with mean μ0 against the alternative hypothesis that the distribution does not have mean μ0.
OneSampleTTest(vec(xs), vec(ys), μ0)

#-----------------
# Population details:
# parameter of interest:   Mean
# value under h_0:         0.560586
# point estimate:          -0.0159316
# 95% confidence interval: (-0.02962, -0.002245)

# Test summary:
# outcome with 95% confidence: reject h_0
# two-sided p-value:           <1e-07

# Details:
# number of observations:   5
# t-statistic:              -116.94897953108811
# degrees of freedom:       4
# empirical standard error: 0.004929653959457991
