import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        s=0
        z_max=np.max(z)
        for i in range(np.size(z)):
            s=s+(np.exp((z[i])-z_max))
        z=np.exp(z-z_max)/s
        return np.round(z, 4)
        pass
