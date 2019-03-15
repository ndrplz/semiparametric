import numpy as np


# List of keypoint names to map the kpoint dictionary into the kpoint array
pascal_kpoint_names = {
    'car': ['left_front_wheel', 'left_back_wheel', 'right_front_wheel',
            'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield',
            'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light',
            'right_front_light', 'left_back_trunk', 'right_back_trunk'],
    'chair': ['back_upper_left', 'back_upper_right', 'seat_upper_left',
              'seat_upper_right', 'seat_lower_left', 'seat_lower_right',
              'leg_upper_left', 'leg_upper_right', 'leg_lower_left',
              'leg_lower_right']
}

pascal_kpoint_to_idx = {
    'car': dict(zip(pascal_kpoint_names['car'], np.arange(len(pascal_kpoint_names['car'])))),
    'chair': dict(zip(pascal_kpoint_names['chair'], np.arange(len(pascal_kpoint_names['chair']))))
}

pascal_idx_to_kpoint = {
    'car': {v: k for (k, v) in pascal_kpoint_to_idx['car'].items()},
    'chair': {v: k for (k, v) in pascal_kpoint_to_idx['chair'].items()}
}

# Colors for 3D model segmentation
pascal_parts_colors = {
    'car': {
        'car': (0.2, 0.2, 0.2),
        'wheel': (1, 0, 0),
        'front-glass': (0, 1, 0),
        'side-glass': (0, 0, 1),
        'rear-glass': (1, 1, 0),
        'radiator': (0, 1, 1),
        'hood': (1, 1, 1),
        'roof': (0.8, 0.25, 0),
        'front-light': (0.5, 0.5, 1),
        'back-light': (1, 0.5, 0.5),
        'back-plate': (0.7, 0.7, 0.7)
    },
}

pascal_stick_planes = {
    'car': {
        'left': [('left_back_trunk', 'left_back_wheel'),
                 ('left_back_wheel', 'left_front_wheel'),
                 ('left_front_wheel', 'left_front_light'),
                 ('left_front_light', 'upper_left_windshield'),
                 ('upper_left_windshield', 'upper_left_rearwindow'),
                 ('upper_left_rearwindow', 'left_back_trunk')],
        'right': [('right_back_trunk', 'right_back_wheel'),
                  ('right_back_wheel', 'right_front_wheel'),
                  ('right_front_wheel', 'right_front_light'),
                  ('right_front_light', 'upper_right_windshield'),
                  ('upper_right_windshield', 'upper_right_rearwindow'),
                  ('upper_right_rearwindow', 'right_back_trunk')],
        'wheel': [('left_back_wheel', 'left_front_wheel'),
                  ('left_front_wheel', 'right_front_wheel'),
                  ('right_front_wheel', 'right_back_wheel'),
                  ('right_back_wheel', 'left_back_wheel')],
        'light': [('left_back_trunk', 'left_front_light'),
                  ('left_front_light', 'right_front_light'),
                  ('right_front_light', 'right_back_trunk'),
                  ('right_back_trunk', 'left_back_trunk')],
        'roof': [('upper_left_rearwindow', 'upper_left_windshield'),
                 ('upper_left_windshield', 'upper_right_windshield'),
                 ('upper_right_windshield', 'upper_right_rearwindow'),
                 ('upper_right_rearwindow', 'upper_left_rearwindow')]
    },
    'chair': {
        'back': [('seat_upper_left', 'back_upper_left'),
                 ('back_upper_left', 'back_upper_right'),
                 ('back_upper_right', 'seat_upper_right'),
                 ('seat_upper_right', 'seat_upper_left')],
        'left': [('leg_lower_left', 'seat_lower_left'),
                 ('seat_lower_left', 'seat_upper_left'),
                 ('seat_upper_left', 'back_upper_left'),
                 ('back_upper_left', 'leg_upper_left')],
        'right': [('leg_lower_right', 'seat_lower_right'),
                  ('seat_lower_right', 'seat_upper_right'),
                  ('seat_upper_right', 'back_upper_right'),
                  ('back_upper_right', 'leg_upper_right')],
        'seat': [('seat_upper_left', 'seat_upper_right'),
                 ('seat_upper_right', 'seat_lower_left'),
                 ('seat_lower_left', 'seat_lower_right'),
                 ('seat_lower_right', 'seat_upper_left')],
    }
}

pascal_texture_planes = {
    'car': {
        'left': ['left_back_trunk', 'left_back_wheel', 'left_front_wheel',
                 'left_front_light', 'upper_left_windshield',
                 'upper_left_rearwindow'],
        'right': ['right_back_trunk', 'right_back_wheel', 'right_front_wheel',
                  'right_front_light', 'upper_right_windshield',
                  'upper_right_rearwindow'],
        'roof': ['upper_left_rearwindow', 'upper_left_windshield',
                 'upper_right_windshield', 'upper_right_rearwindow'],
        'front': ['left_front_light', 'right_front_light',
                  'upper_right_windshield', 'upper_left_windshield'],
        'back': ['left_back_trunk', 'right_back_trunk',
                 'upper_right_rearwindow', 'upper_left_rearwindow']
    },
    'chair': {
        'left': ['leg_lower_left', 'seat_lower_left', 'seat_upper_left',
                 'back_upper_left', 'leg_upper_left'],
        'right': ['leg_lower_right', 'seat_lower_right', 'seat_upper_right',
                  'back_upper_right', 'leg_upper_right'],
        'seat': ['seat_lower_left', 'seat_upper_left',
                 'seat_upper_right', 'seat_lower_right'],
        'back': ['seat_upper_left', 'back_upper_left',
                 'back_upper_right', 'seat_upper_right']
    }
}
