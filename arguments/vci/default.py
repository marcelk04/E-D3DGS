ModelParams = dict(
	sh_degree = 3,
	eval = True,
	render_process = False,
	loader = "vci",
	shuffle = True,
)

ModelHiddenParams = dict(
	net_width = 128,
	defor_depth = 1,
	min_embeddings = 30,
	max_embeddings = 150,
	no_ds = False,
	no_dr = False,
	no_do = False,
	no_dc = False,
	
	temporal_embedding_dim = 256,
	gaussian_embedding_dim = 32,
	use_coarse_temporal_embedding = True,
	no_c2f_temporal_embedding = False,
	no_coarse_deform = False,
	no_fine_deform = False,
	
	total_num_frames = 325,
	c2f_temporal_iter = 10000,
	deform_from_iter = 0,
	use_anneal = False,
	zero_temporal = True,
)

OptimizationParams = dict(
	dataloader = True,
	iterations = 40_000,
	maxtime = 325,
	batch_size = 1,
	
	position_lr_init = 0.00004,
	position_lr_final = 0.0000004,
	position_lr_delay_mult = 0.01,
	position_lr_max_steps = 40_000,

	deformation_lr_init = 0.000016,
	deformation_lr_final = 0.0000016,
	deformation_lr_delay_mult = 0.01,
	deformation_lr_max_steps = 40_000,
	
	feature_lr = 0.0025,
	feature_lr_div_factor = 20.0,
	opacity_lr = 0.05,
	scaling_lr = 0.005,
	rotation_lr = 0.001,
	
	percent_dense = 0.01,
	lambda_dssim = 1.0,
	lambda_lpips = 0,
	
	weight_constraint_init = 1,
	weight_constraint_after = 0.2,
	weight_decay_iteration = 5000,

	densification_interval = 100,
	densify_from_iter = 5000,
	densify_until_iter = 60_000,
	densify_grad_threshold_fine_init = 0.0003,
	densify_grad_threshold_after = 0.0003,
	pruning_from_iter = 20000,
	pruning_interval = 1000,

	opacity_reset_interval = 6000000,
	opacity_threshold_fine_init = 0.005,
	opacity_threshold_fine_after = 0.005,
	reset_opacity_ratio = 0,
	opacity_l1_coef_fine = 0.0001,
	
	scene_bbox_min = [-2.5, -2.0, -1.0],
	scene_bbox_max = [2.5, 2.0, 1.0],
	num_pts = 2000,
	threshold = 3,
	downsample = 1.0,
	
	use_dense_colmap = False,
	use_colmap = True,
	coef_tv_temporal_embedding = 0.0001,
	random_until = 40000,
	num_multiview_ssim = 2,
	offsets_lr = 0.00002,
	reg_coef = 1.0,
)