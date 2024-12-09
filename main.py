import sys
import logging
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from models.Vgae import *
from utils.dataloader import *
from utils.functions import *
from utils.evaluation import *
from models.KMeans_fun import *
from models.MLP_model import MLP
from models.diffusion_model import *
from models.LightGCN import LGCN_Encoder
from utils.loss_functions import *

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Coach:
    def __init__(self, dataloader):
        self.LightGCN_optimizer = None
        self.LightGCN_model = None
        self.train_graph_data = dataloader['train']
        self.test_graph_data = dataloader['test']
        self.node_features, self.edge_index, self.num_nodes = prepare_data(self.train_graph_data, device)

        self.train_interaction_matrix = get_train_user_item_matrix(self.train_graph_data)
        self.interaction_matrix, self.num_users, self.num_items = get_user_item_matrix(self.train_graph_data)
        self.normalized_adj_matrix = normalize_graph_mat(self.interaction_matrix)

        generate_interaction_group_dict(self.train_interaction_matrix, args.k)

        print(self.num_users)
        print(self.num_items)

        self.VGAE_model = Vgae(64, 64, 64, device, self.num_nodes).to(device)
        self.Diffusion_model = DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min,
                                                args.noise_max, args.steps, device).to(device)

        output_dimensions = [args.dims] + [args.n_hid]
        input_dimensions = output_dimensions[::-1]
        self.MLP_model = MLP(input_dimensions, output_dimensions, args.emb_size, time_type="cat", norm=args.norm).to(
            device)
        self.LightGCN_model = LGCN_Encoder(self.num_users, 2).to(device)

        self.VGAE_optimizer = torch.optim.Adam([{'params': self.VGAE_model.parameters(), 'weight_decay': 0}],
                                               lr=args.lr)
        self.LightGCN_optimizer = torch.optim.Adam([{'params': self.LightGCN_model.parameters(), 'weight_decay': 0}],
                                                   lr=0.001)

        self.MLP_optimizer = torch.optim.Adam([{'params': self.MLP_model.parameters(), 'weight_decay': 0}], lr=args.lr)


        self.batch_dataset = DRO_dataloader(self.train_interaction_matrix, self.num_users, self.num_items)
        self.dataloader = DataLoader(self.batch_dataset, batch_size=args.batch_size, shuffle=True)

    def train_model(self):
        best_performance = []
        evaluation_results = {}
        loss_group = torch.zeros(args.k).to(device)
        weight_list = [torch.ones(1).to(device) for _ in range(args.k)]
        loss_list = [0 for _ in range(args.k)]

        train_edge_idx = mask_test_edges_dgl(self.train_graph_data)
        train_graph = dgl.edge_subgraph(self.train_graph_data, train_edge_idx, relabel_nodes=False).to(device)
        adj_matrix = train_graph.adjacency_matrix().to_dense().to(device)
        weight_tensor, normalization = compute_loss_para(adj_matrix, device)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

        log_save = './logs/'

        current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        log_file = args.save_name
        fname = f'{log_file}_{current_time}.txt'

        log_dir = os.path.join(log_save, args.dataset)
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, fname)

        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger()
        self.logger.addHandler(fh)

        logging.Formatter.converter = time.localtime

        self.logger.info(args)
        self.logger.info('================')

        # Initialize the KL divergence weight and starting temperature
        initial_kl_weight = 1.0  # Initial KL weight
        final_kl_weight = 0.01  # Final KL weight
        temperature = 0.02  # Temperature parameter
        cooling_rate = 0.001  # Cooling rate

        for epoch in range(args.epoch):
            average_loss = 0
            count = 0

            # Generate embeddings
            batch_latent = self.VGAE_model.encoder(self.node_features, self.edge_index)
            diffusion_terms = self.Diffusion_model.caculate_losses(self.MLP_model, batch_latent, args.reweight)
            logits = self.VGAE_model.decoder(diffusion_terms["pred_xstart"])
            elbo_loss = diffusion_terms["loss"].mean()

            vgae_loss = compute_vgae_loss(logits, adj_matrix, normalization, self.VGAE_model, weight_tensor)

            # Dynamically adjust the KL divergence weight
            kl_weight = initial_kl_weight * (final_kl_weight / initial_kl_weight) ** (epoch / args.epoch)

            # Calculate the total loss with weighting
            total_pretrain_loss = elbo_loss + kl_weight * vgae_loss

            # Update the temperature after each epoch
            temperature *= cooling_rate  # Temperature for the next update

            # for epoch in range(args.epoch):
        #     average_loss = 0
        #     count = 0
        #
        #     # generating embedding
        #     batch_latent = self.VGAE_model.encoder(self.node_features, self.edge_index)
        #     diffusion_terms = self.Diffusion_model.caculate_losses(self.MLP_model, batch_latent, args.reweight)
        #     logits = self.VGAE_model.decoder(diffusion_terms["pred_xstart"])
        #     elbo_loss = diffusion_terms["loss"].mean()
        #
        #     vgae_loss = compute_vgae_loss(logits, adj_matrix, normalization, self.VGAE_model, weight_tensor)
        #
        #     total_pretrain_loss = elbo_loss + vgae_loss
            torch.cuda.empty_cache()
            self.VGAE_optimizer.zero_grad()
            self.MLP_optimizer.zero_grad()
            total_pretrain_loss.backward()
            self.VGAE_optimizer.step()
            self.MLP_optimizer.step()

            embeddings_list = []

            with torch.no_grad():
                vgae_embeddings = self.VGAE_model.encoder(self.node_features, self.edge_index)
                denoised_embeddings = self.Diffusion_model.p_sample(self.MLP_model, vgae_embeddings,
                                                                    args.sampling_steps,
                                                                    args.sampling_noise)
                embeddings_list.append(denoised_embeddings)

            combined_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)

            user_embeddings = combined_embeddings[:self.num_users]
            item_embeddings = combined_embeddings[self.num_users:]

            for batch in tqdm(self.dataloader):
                all_embeddings = self.LightGCN_model(self.normalized_adj_matrix, user_embeddings,
                                                     item_embeddings).to(device)
                ideal_dist = ideal_distribution_cal(all_embeddings, args.dataset)

                user_ids, pos_item_ids, neg_item_ids, batch_stage = [x.to(device) for x in batch]

                user_emb = all_embeddings[user_ids]
                pos_item_emb = all_embeddings[pos_item_ids]
                neg_item_emb = all_embeddings[neg_item_ids]

                reconstruction_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + \
                                      l2_reg_loss(1e-3, user_emb, pos_item_emb, neg_item_emb)

                for group_idx in range(args.k):
                    indices = (batch_stage == group_idx)
                    single_loss = torch.sum(reconstruction_loss * (indices).cuda())
                    loss_group[group_idx] = single_loss

                performance_terms = torch.tensor([torch.sum(batch_stage == g_idx) for g_idx in range(args.k)]).cuda()
                total_loss = torch.sum(loss_group, dim=0)
                rec_loss_list = total_loss / (performance_terms + 1e-16)

                total_loss_value = rec_loss_list

                for i in range(args.k):
                    if len(torch.nonzero(batch_stage == i)) == 0:
                        loss_per_group = loss_list[i]
                    else:
                        loss_per_group = total_loss_value[i]

                    group_indices = torch.nonzero(batch_stage == i).squeeze()

                    if group_indices.numel() != 0:
                        group_embedding = all_embeddings[group_indices]
                        if group_embedding.dim() == 1:
                            group_embedding = group_embedding.unsqueeze(0)

                        sinkhorn_loss_value = sinkhorn_distance(group_embedding, ideal_dist)

                        log_group_embedding = torch.log(group_embedding)
                        log_group_embedding = torch.nan_to_num(log_group_embedding, nan=0.0)

                        weighted_group_embedding = group_embedding.mul(log_group_embedding)
                        sum_weighted_embedding = weighted_group_embedding.sum()
                        mean_weighted_embedding = sum_weighted_embedding.mean()

                        total_group_loss = (
                                loss_per_group - args.sinkhorn_weight * sinkhorn_loss_value - 0.001 * mean_weighted_embedding)
                    else:
                        total_group_loss = loss_per_group

                    loss_list[i] = (1 - 0.3) * loss_list[i] + 0.3 * total_group_loss

                    update_factor = args.step_size * loss_list[i]
                    if isinstance(update_factor, float):
                        update_factor = torch.tensor(update_factor)
                    weight_list[i] = weight_list[i] * torch.exp(update_factor).clone()

                sum_weights = sum(weight_list)
                weight_list = [i / sum_weights for i in weight_list]

                final_loss = torch.zeros(1).to(device)
                for i in range(args.k):
                    final_loss += weight_list[i] * loss_list[i]

                average_loss += final_loss
                count += 1
                self.LightGCN_optimizer.zero_grad()
                final_loss.backward()
                self.LightGCN_optimizer.step()

                weight_list = [i.detach() for i in weight_list]
                # loss_list = [i.detach() for i in loss_list]
                loss_list = [torch.tensor(i).detach() if isinstance(i, float) else i.detach() for i in loss_list]
                loss_group = loss_group.detach()

            print(f'Epoch {epoch} group loss: {loss_list}')
            print(f'Epoch {epoch} group weights: {weight_list}')
            average_loss = average_loss / count
            print(f'EPOCH[{epoch + 1}/{args.epoch}] {average_loss.item()}')

            measure, best_epoch = self.test_model(epoch, best_performance)
            evaluation_results[epoch] = measure
            torch.cuda.empty_cache()

        self.logger.info('The best result of %s:\n%s' % ('SDRO', ''.join(evaluation_results[best_epoch - 1])))

    def test_model(self, epoch, best_performance):
        self.MLP_model.eval()
        self.LightGCN_model.eval()
        embeddings_list = []
        test_feats, test_edge_index, _ = prepare_data(self.test_graph_data, device)

        with torch.no_grad():
            vgae_embeddings = self.VGAE_model.encoder(test_feats, test_edge_index)
            denoised_embeddings = self.Diffusion_model.p_sample(self.MLP_model, vgae_embeddings, args.sampling_steps,
                                                                args.sampling_noise)
            embeddings_list.append(denoised_embeddings)

        all_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)
        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]
        test_interaction_matrix = generate_interaction_matrix_from_dgl(self.test_graph_data, self.num_users,
                                                                       self.num_items)
        test_norm_adj_matrix = normalize_graph_mat(test_interaction_matrix)
        # model = LGCN_Encoder(self.num_users, 3, test_norm_adj_matrix, user_embeddings, item_embeddings)
        # rec_model = model.to(device)

        with torch.no_grad():
            all_embeddings = self.LightGCN_model(test_norm_adj_matrix, user_embeddings,
                                                 item_embeddings)
            user_embeddings = all_embeddings[:self.num_users]
            item_embeddings = all_embeddings[self.num_users:]

        scores = torch.matmul(user_embeddings, item_embeddings.t())
        origin_interactions, user_set = get_origin_user_interaction_list(self.test_graph_data, self.num_users)
        rec_dict = get_rec_list(user_set, scores, self.num_users)

        measure = ranking_evaluation(origin_interactions, rec_dict, [10, 20])
        measure_index = measure.index('Top 20\n')
        measure_input = measure[measure_index:]
        best_epoch = fast_evaluation(epoch, measure_input, best_performance, self.logger)

        print(best_epoch[0])

        return measure, best_epoch[0]


if __name__ == "__main__":
    seed_it(1024)
    dataset = load_datasets(args.dataset)
    coach_instance = Coach(dataset)
    coach_instance.train_model()
