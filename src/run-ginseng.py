import visualization
from src import util

# region load images
ir_image = util.load_ir_in_dat('./ginseng-ir/irimage_20220728_0903.dat')
rgb_image = util.load_rgb_in_jpeg('./ginseng-rgb/1.jpeg')
# endregion

exg = util.get_excess_green(rgb_image)
label = util.get_leaf_by_jenks(exg)
predict = util.get_leaf_by_jenks(ir_image)

visualization.show_image(ir_image, 'IR image')
visualization.show_image(rgb_image, 'RGB image')
visualization.show_histogram(ir_image, "IR histogram")
visualization.show_image(exg, 'EXG image')
visualization.show_image(label, 'Label image')
visualization.show_image(predict, 'Predict image')

accuracy = util.get_intersection_over_union(label, predict)
print(f"정확도(IOU, %): {accuracy*100:.2f}")

max_temperature = util.get_max_temperature(ir_image, predict)
average_temperature = util.get_average_temperature(ir_image, predict)
min_temperature = util.get_min_temperature(ir_image, predict)

# TODO: not works because pixel's range in byte
print(f"잎의 최고 온도(C): {max_temperature:.2f}")
print(f'잎의 평균 온도(C): {average_temperature:.2f}')
print(f"잎의 최저 온도(C): {min_temperature:.2f}")
