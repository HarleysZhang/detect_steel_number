# import json
# import os
#
# annotations = json.load(open("gangjin_mask_train.json"))
# annotations = list(annotations.values())  # don't need the dict keys
# print(annotations)


import json
import pandas as pd
import skimage

coco = dict()


def parseCsvFiles(csv_path):
    csv_file = pd.read_csv(csv_path)
    last_image_id = ''
    per_image_data = dict()
    box_count = 0
    for i in range(len(csv_file)):
        csv_loc = csv_file.iloc[i]
        image_id = csv_loc['ID']
        image_detection = csv_loc[' Detection']
        image_detection = list(image_detection.split())
        image_detection = [int(det) for det in image_detection]
        x1, y1, x2, y2 = image_detection
        all_points_x = [x1, x1, x2, x2, x1]
        all_points_y = [y1, y2, y2, y1, y1]
        if image_id != last_image_id:
            if i != 0:
                coco[last_image_id] = per_image_data
            last_image_id = image_id
            box_count = 0
            per_image_data = dict()
            per_image_data['fileref'] = ''
            image = skimage.io.imread('../train_dataset/'+image_id)
            height, width = image.shape[:2]
            per_image_data['size'] = int(height*width)
            per_image_data['filename'] = image_id
            per_image_data['base64_img_data'] = ''
            per_image_data['file_attributes'] = dict()
            per_image_data['regions'] = dict()
        per_box_data = dict()
        per_box_data_shape = dict()
        per_box_data_region = dict()
        per_box_data_shape['name'] = 'polygon'
        per_box_data_shape['all_points_x'] = all_points_x
        per_box_data_shape['all_points_y'] = all_points_y
        per_box_data['shape_attributes'] = per_box_data_shape
        per_box_data['region_attributes'] = per_box_data_region
        per_image_data['regions'][str(box_count)] = per_box_data
        box_count += 1


if __name__ == '__main__':
    csv_path = '../train_labels.csv'
    json_file = './gangjin_mask_train.json'
    parseCsvFiles(csv_path)
json.dump(coco, open(json_file, 'w'))
