import os
import json
from cfg import *

with open(LABEL_FILE, "w+") as f:
    for json_file in os.listdir("data/outputs"):
        with open(os.path.join("data/outputs", json_file), "r") as jf:
            j = json.load(jf)
            print(j)
            pic = j['path'].split("\\")[-1]
            boxes = j['outputs']['object']
            wh = j['size']
            w, h = wh['width'], wh['height']

            w_scale, h_scale = w / IMG_WIDTH, h / IMG_HEIGHT

            f.write(pic)
            for box in boxes:
                bndbox = box['bndbox']
                _x1, _y1, _x2, _y2 = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
                _w, _h = _x2 - _x1, _y2 - _y1

                _w0_5, _h0_5 = _w / 2, _h / 2
                _cx, _cy = _x1 + _w0_5, _y1 + _h0_5
                x1, y1, w, h = int(_cx / w_scale), int(_cy / h_scale), int(_w / w_scale), int(_h / h_scale)
                f.write(" {} {} {} {} {}".format(int(box['name']), x1, y1, w, h))

            f.write("\n")
            f.flush()
