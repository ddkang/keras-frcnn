import os
import json

def get_data(input_path):
    all_imgs = []
    # For MS-COCO, we refer to the classes by their IDs
    classes_count = {key: 0 for key in range(1, 91)}

    for json_fname, imageset in [('instances_val2014.json', 'val'),
                                 ('instances_train2014.json', 'train')]:
        def parse_images(x):
            image_dir = os.path.join('images', imageset + '2014')
            fname = os.path.join(input_path, image_dir, x['file_name'])
            return (x['id'],
                    {'filepath': fname,
                     'width': x['width'],
                     'height': x['height'],
                     'bboxes': [],
                     'imageset': imageset})
        def parse_annotations(x, img_data):
            tmp = {'class': x['category_id'],
                   'x1': x['bbox'][0],
                   'x2': x['bbox'][0] + x['bbox'][1],
                   'y1': x['bbox'][2],
                   'y2': x['bbox'][2] + x['bbox'][3]}
            classes_count[x['category_id']] += 1
            img_data[x['image_id']]['bboxes'].append(tmp)


        fname = os.path.join(input_path, 'annotations', json_fname)
        with open(fname) as f:
            data = json.load(f)

        img_data = dict(map(parse_images, data['images']))
        for anno in data['annotations']:
            parse_annotations(anno, img_data)

        for key in img_data:
            all_imgs.append(img_data[key])

    for c in classes_count.keys():
        if classes_count[c] == 0:
            del classes_count[c]
    class_mapping = {}
    for i, c in enumerate(sorted(classes_count.keys())):
        class_mapping[c] = i

    return all_imgs, classes_count, class_mapping
