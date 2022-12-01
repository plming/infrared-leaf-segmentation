import os

dir_ir = os.listdir('./ginseng-ir')
dir_rgb = os.listdir('./ginseng-rgb')

ir_timestamps = {os.path.splitext(file)[0].split('_', maxsplit=1)[1] for file in dir_ir}
rgb_timestamps = {os.path.splitext(file)[0].split('_', maxsplit=1)[1] for file in dir_rgb}

intersection = ir_timestamps.intersection(rgb_timestamps)

# remove files that doesn't contain intersection's timestamp
for file in dir_ir:
    filename = os.path.splitext(file)[0]
    timestamp = filename.split('_', maxsplit=1)[1]
    if timestamp not in intersection:
        os.remove(os.path.join('./ginseng-ir', file))

for file in dir_rgb:
    filename = os.path.splitext(file)[0]
    timestamp = filename.split('_', maxsplit=1)[1]
    if timestamp not in intersection:
        os.remove(os.path.join('./ginseng-rgb', file))

assert len(os.listdir('./ginseng-ir')) == len(os.listdir('./ginseng-rgb'))
