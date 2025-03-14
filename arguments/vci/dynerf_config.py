ModelParams = dict(
    loader = "vci"
)

ModelHiddenParams = dict(
    defor_depth = 1,
    net_width = 128,
    no_ds = False,
    no_do = False,
    no_dc = False,
    
    use_coarse_temporal_embedding = True,
    c2f_temporal_iter = 10000,
    deform_from_iter = 5000,
    total_num_frames = 325,
)

OptimizationParams = dict(
    dataloader = True,
    batch_size = 1,
    iterations = 40_000,
    maxtime = 325,

    densify_from_iter = 5000,    
    pruning_from_iter = 5000,

    densify_grad_threshold_fine_init = 0.0003,
    densify_grad_threshold_after = 0.0003,

    opacity_threshold_fine_init = 0.0001,
    opacity_threshold_fine_after = 0.0001,
    
    densify_until_iter = 40_000,
    position_lr_max_steps = 40_000,
    deformation_lr_max_steps = 40_000,

    lambda_dssim = 0,
    num_multiview_ssim = 0,
    use_colmap = True,
    reg_coef = 0,
)