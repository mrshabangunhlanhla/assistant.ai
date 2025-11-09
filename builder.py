import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

img = cv2.imread("assets/image.png")
cv2.resize(500, 6)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# z = np.zeros([10, 10])
# x = np.linspace(-2*math.pi, 2*math.pi, 100)
# def f(x):
#     return np.sin((x/(1)))

# y = f(x)

# plt.xlabel("X-axis Label")
# plt.ylabel("Y-axis Label")
# plt.xlim(-2*math.pi, 2*math.pi)
# plt.ylim(-2, 2)
# # print(x.shape, )
# plt.legend()
# plt.grid(True)
# # print(x, y)
# plt.plot(x, y)
# plt.show()



# def great(name='bob'):
#     print(f"""Hello {name} """)

# great("bob")