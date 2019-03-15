import argparse
from numpy.random import choice


def random_search(net_name: str, args: argparse.Namespace):
    """
    Perform random hyper-parameter search on args and returns
      the newly sampled arguments.
    """
    if net_name == 'vunet':
        lr_values = [1e-5, 5*1e-5, 1e-4, 2*1e-4, 5*1e-4]
        w_norm_values = [True, False]
        drop_prob_values = [0.0, 0.1, 0.15, 0.2, 0.25, 0.5]

        w_L_values = [1, 3, 5, 10, 15, 20, 25]
        w_AB_values = [1, 3, 5, 10, 15, 20, 25, 50, 60, 80, 100]
        w_content_values = [1, 3, 5, 10, 15, 20, 25, 50, 60, 80, 100]
        w_KL_values = [0.0, 0.1, 0.5, 1, 5, 10, 20, 50, 80]

        shaded_values = [True, False]
        shaded_segm_values = [True, False]

        vgg_pool_values = ['avg', 'max']
        
        args.w_AB = float(choice(w_AB_values))
        args.w_L = float(choice(w_L_values))
        args.w_content = float(choice(w_content_values))
        args.w_KL = float(choice(w_KL_values))
        
        args.shaded = choice(shaded_values)
        args.shaded_segm = choice(shaded_segm_values)

        args.w_norm = choice(w_norm_values)

        args.drop_prob = float(choice(drop_prob_values))

        args.lr = float(choice(lr_values))

        args.vgg_pool = choice(vgg_pool_values)
        
    else:
        raise NotImplementedError()

    return args
