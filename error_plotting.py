import numpy as np
import matplotlib.pyplot as plt
from math import log2


################ Error with equally devided rectangle ###################
L2_error = [0.007216210382791447, 0.006006678559238461, 0.0016931431430956877, 0.0004402533159348543,
            0.00011156859206633449, 2.8020113260487367e-05, 7.015499966188221e-06, 1.7547061166167832e-06,
            4.387408889375761e-07, 1.0969010596838715e-07]
H10_error= [0.06691503335772368, 0.059179558257117064, 0.031779121513786104, 0.016219210294881863,
            0.008159521525763264, 0.004087086434995247, 0.002044593564627502, 0.0010224448477584363,
            0.0005112430262737956, 0.00025562435013193286]

n_element = [4**i for i in range(1,len(L2_error) + 1)]

ax1 = plt.subplot(2, 2, 1)
ax1.set_title("L2 error")
ax1.plot(n_element, L2_error, 'bo-')

ax2 = plt.subplot(2, 2, 2)
ax2.set_title("H10 error")
ax2.plot(n_element, H10_error, 'ro-')

print('L2 eoc plot')
l2_eoc = [log2(L2_error[i]/L2_error[i+1]) for i in range(0, len(L2_error) - 2)]
h10_eoc = [log2(H10_error[i]/H10_error[i+1]) for i in range(0, len(H10_error) - 2)]
iter = [i for i in range(1,len(l2_eoc)+ 1)]

ax3 = plt.subplot(2, 2, 3)
ax3.set_title("L2 eoc")
ax3.plot(iter, l2_eoc, 'bo-')

ax4 = plt.subplot(2, 2, 4)
ax4.set_title("H10 eoc")
ax4.plot(iter, h10_eoc, 'ro-')

plt.show()