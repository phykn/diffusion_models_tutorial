from .ddpm import DDPM

def build_model(args):
    return DDPM(
        dim=args.dim,
        init_dim=args.init_dim,
        out_dim=args.out_dim,
        dim_mults=args.dim_mults,
        channels=args.channels,
        self_condition=args.self_condition,
        resnet_block_groups=args.resnet_block_groups,
        steps=args.steps,
        device=args.device
    )