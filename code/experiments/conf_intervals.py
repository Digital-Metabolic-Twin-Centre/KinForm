"""
Only KinForm-L
Train 5 kGroupFold. Each time save predictions of each tree on each point in the test set.
- Compute confidence intervals for the predictions (mean +/- sd)
- Calc accuracy (percentage of real values that fall within the confidence intervals)
- Show accuracy per similarity bin
- How big are confidence intervals? Visualize
- How big are confidence intervals within each similarity bin? Visualize
... more
"""