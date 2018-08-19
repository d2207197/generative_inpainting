import cv2
from dramatiq import group
from models import m0, m1, m2
# raw_image = cv2.imread('raw_image.jpg')
raw_image = '/home/iamifengc/pixnet-2018/generative_inpainting/raw_image.jpg'
m0_pred = m0.predict.send(raw_image).get_result(block=True, timeout=20000)
m1_pred = m1.predict.sent(raw_image).get_result(block=True, timeout=20000)
m2_pred = m2.predict.sent(raw_image).get_result(block=True, timeout=20000)

# m0_pred = m0.predict.send(raw_image).get_result(block=True)
# m1_pred = m1.predict.send(raw_image).get_result(block=True)

# g = group([
#     m0.predict.send(raw_image),
#     m1.predict.send(raw_image),
#     m2.predict.send(raw_image),
# ]).run()
# g.wait(timeout=10_000)
# for res in g.get_results(block=True, timeout=5_000):
# 	print(res)
