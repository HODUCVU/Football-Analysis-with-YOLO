def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox 
    # Get center
    centerx = (x1 +x2)//2 
    centery = (y1 +y2)//2 
    return int(centerx), int(centery)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def get_bbox_height(bbox):
    return bbox[3] - bbox[1]
