def car_planes_visibility(pascal_class: str, az_deg: int):
    """
    Get the list of planes which are visible given the current viewpoint
    """
    visible = []
    if pascal_class == 'car':
        if az_deg == 0:
            visible.append('front')
        elif 0 < az_deg < 90:
            visible.extend(['front', 'left'])
        elif az_deg == 90:
            visible.append('left')
        elif 90 < az_deg < 180:
            visible.extend(['left', 'back'])
        elif az_deg == 180:
            visible.append('back')
        elif 180 < az_deg < 270:
            visible.extend(['right', 'back'])
        elif az_deg == 270:
            visible.append('right')
        elif 270 < az_deg < 360:
            visible.extend(['right', 'front'])
    # TODO CHECK!!!
    elif pascal_class == 'chair':
        if az_deg == 0:
            visible.extend(['back', 'seat'])
        elif 0 < az_deg < 90:
            visible.extend(['back', 'seat', 'left'])
        elif az_deg == 90:
            visible.extend(['left'])
        elif 90 < az_deg < 180:
            visible.extend(['left', 'back'])
        elif az_deg == 180:
            visible.append('back')
        elif 180 < az_deg < 270:
            visible.extend(['right', 'back'])
        elif az_deg == 270:
            visible.append('right')
        elif 270 < az_deg < 360:
            visible.extend(['back', 'seat', 'right'])
    else:
        raise NotImplementedError()
    return visible
