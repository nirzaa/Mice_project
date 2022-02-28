import numpy as np
nums = [1,2,3,2]

for i in range(1, len(nums)):
    occr = np.count_nonzero(nums == i)
    if occr > 1:
        dupl = i
    if occr == 0:
        missing = i
result = [dupl, missing]
print(result)