import argparse

parser = argparse.ArgumentParser(description="Go S_DRO")

parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=4096, help="the batch size for bpr loss training procedure")
parser.add_argument('--dataset', type=str, default='kuairec', help="[yelp2018, douban, kuairec]")
parser.add_argument('--epoch', type=int, default=5, help="the number of training epochs")
parser.add_argument('--decay', type=float, default=1e-3)
parser.add_argument('--n_hid', type=int, default=64)

# params for the denoiser
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=int, default=64, help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=16, help='timestep embedding size')

# params for diffusions
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True,
                    help='assign different weight to different timestep or not')

parser.add_argument('--p', type=float, default=1)
parser.add_argument('--k', type=int, default=5, help='the number of clusters')
parser.add_argument("--sinkhorn_weight", type=float, default=1)
parser.add_argument('--step_size', type=float, default=0.001)

parser.add_argument('--save_name', type=str, default='tem')

args = parser.parse_args()
