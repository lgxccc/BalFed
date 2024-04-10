import random
from train_single import MatrixLoader
import numpy as np
import torch
from tqdm import tqdm
from ncf_model import NeuralCollaborativeFiltering
from dataloader import MovielensDatasetLoader
from ncf_server_model import ServerNeuralCollaborativeFiltering
from svdpp_model import SVDPP
from svdpp_server_model import ServerSVDPP
from train_single import NCFTrainer
from utils import Utils, seed_everything
from metrics import compute_metrics,compute_metrics_5
from deepFM_server import ServerDeepFM
from deepFM import DeepFM
import json
import argparse
import logging
class FederatedNCF:
    def __init__(self,
                 data_loader: MovielensDatasetLoader,
                 num_clients=50,
                 user_per_client_range=(1, 5),
                 model="ncf",
                 aggregation_epochs=50,
                 local_epochs=10,
                 batch_size=128,
                 latent_dim=32,
                 lr=0.001,
                 client_data_thresh=1,
                 thresh=1,
                 seed=0,
                 lower_lamda=1,
                 higher_lamda=1,
                 mean_num=1,
                 aggregation='fedavg'
                 ):
        self.seed = seed
        seed_everything(seed)
        self.data_loader = data_loader
        self.test_set = data_loader.latest_ratings
        self.ui_matrix = self.data_loader.ratings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.user_per_client_range = user_per_client_range
        self.lower_lamda = lower_lamda
        self.higher_lamda = higher_lamda
        self.mean_num = mean_num
        self.model = model
        self.aggregation = aggregation
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_data_thresh = client_data_thresh
        self.thresh = thresh
        self.clients = self.generate_clients()
        self.client_nums = [client.nonzero for client in self.clients]
        self.model_optimizers = [torch.optim.Adam(client.model.parameters(), lr=lr) for client in self.clients]
        self.utils = Utils(self.num_clients)
        self.hrs = []
        self.ndcg = []
        self.loss = []

    def generate_clients(self):
        clients = []
        count_list = []
        self.random_integers = []
        while len(self.random_integers) < self.num_clients:
            num = random.randint(0, len(self.ui_matrix)-1)
            client_matrix = self.ui_matrix[num]
            client_clean_num = np.argwhere(client_matrix >= 1)
            if len(client_clean_num) >= 0 and num not in self.random_integers:
                self.random_integers.append(num)
                count_list.append(len(client_clean_num))
        mean = np.mean(count_list)
        std_dev = np.std(count_list)
        lower_thresh = round(mean - std_dev / self.lower_lamda)
        upper_thresh = round(mean + std_dev * self.higher_lamda)
        for i in self.random_integers:
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(user_ids=list(range(i, i + users)),
                                      data_loader=self.data_loader,
                                      epochs=self.local_epochs,
                                      batch_size=self.batch_size,
                                      model=self.model,
                                      thresh=self.thresh,
                                      lower_thresh=lower_thresh,
                                      upper_thresh=upper_thresh,
                                      mean=mean,
                                      mean_num=self.mean_num,
                                      latent_dim=self.latent_dim))
        return clients

    def single_round(self, epoch=0, grad_mean=None, user_dict=None):
        single_round_results = {key: [] for key in ["num_users", "loss"]}
        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            results,user_dict = client.train(self.model_optimizers[client_id], epoch, user_dict, grad_mean)
            for k, i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"epoch": epoch}
            printing_single_round.update({k: round(sum(i) / len(i), 4) for k, i in single_round_results.items()})
            model = torch.jit.script(client.model.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp" + str(client_id) + ".pt")
            bar.set_description(str(printing_single_round))
        self.loss.append(single_round_results["loss"])
        bar.close()

    def extract_item_models(self):
        for client_id in range(self.num_clients):
            model = torch.jit.load("./models/local/dp" + str(client_id) + ".pt")
            if self.model == 'ncf':
                item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            elif self.model == 'svdpp':
                item_model = ServerSVDPP(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            elif self.model == 'deepfm':
                item_model = ServerDeepFM(self.ui_matrix.shape[1], self.latent_dim)
            else:
                raise ValueError('can not found a server model matched')
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp" + str(client_id) + ".pt")

    def train(self):
        if self.model == 'ncf':
            server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],predictive_factor=self.latent_dim)
            ori_client_model = NeuralCollaborativeFiltering(self.ui_matrix.shape[0], self.ui_matrix.shape[1],self.latent_dim).to(self.device)
        elif self.model == 'svdpp':
            server_model = ServerSVDPP(item_num=self.ui_matrix.shape[1],predictive_factor=self.latent_dim)
            ori_client_model = SVDPP(self.ui_matrix.shape[0], self.ui_matrix.shape[1], self.latent_dim).to(self.device)
        elif self.model == 'deepfm':
            server_model = ServerDeepFM(self.ui_matrix.shape[1], self.latent_dim)
            ori_client_model = DeepFM(self.ui_matrix.shape[0], self.ui_matrix.shape[1], self.latent_dim).to(self.device)
        else:
            raise ValueError('can not found a server model matched')
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server" + str(0) + ".pt")
        ori_client_model.load_server_weights(server_model)
        self.metric_hits = []
        self.metric_ndcg = []
        self.metric_hits_5 = []
        self.metric_ndcg_5 = []
        current_global_prob_list = []
        self.grad_mean = tuple(p.clone() for p in ori_client_model.parameters())

        for epoch in range(self.aggregation_epochs):
            user_dict = {}
            server_model = torch.jit.load("./models/central/server" + str(epoch) + ".pt",
                                          map_location=self.device)
            _ = [client.model.to(self.device) for client in self.clients]
            _ = [client.model.load_server_weights(server_model) for client in self.clients]
            self.grad_mean = self.mean_grad(self.clients)
            self.single_round(epoch=epoch, grad_mean=self.grad_mean, user_dict=user_dict)
            self.extract_item_models()
            self.client_nums = [len(client.loader.positives) for client in self.clients]
            current_local_prob_list = [client.local_pos_prob for client in self.clients]
            current_global_prob = sum(current_local_prob_list) / sum(1 for element in current_local_prob_list if element != 0) if current_local_prob_list else 0
            current_global_prob_list.append(current_global_prob)
            for client in self.clients:
                client.local_pos_prob_scale = client.local_pos_prob / max(current_local_prob_list)
                alpha = 0.8
                mix_rate = np.random.beta(alpha, alpha)
                client.local_thresh = mix_rate * current_global_prob + (1 - mix_rate) * client.local_pos_prob_scale
                client.global_thresh = current_global_prob
            if self.aggregation=="fedavg":
                self.utils.federate(self.client_nums)
            elif self.aggregation=="simpleavg":
                self.utils.simple_avg()
            elif self.aggregation=="fedfast":
                self.utils.fedfast(self.client_nums,self.model)
            else:
                raise ValueError('please select a aggregation algorithm')
            self.test(epoch)
        hits_ave = sum(self.metric_hits) / len(self.metric_hits)
        ndcg_ave = sum(self.metric_ndcg) / len(self.metric_ndcg)
        hits_ave_5 = sum(self.metric_hits_5) / len(self.metric_hits_5)
        ndcg_ave_5 = sum(self.metric_ndcg_5) / len(self.metric_ndcg_5)
        logging.info(f"{hits_ave:.2f},{ndcg_ave:.2f},{hits_ave_5:.2f},{ndcg_ave_5:.2f}")

    def mean_grad(self, sampled_clients):
        grad_sum = [torch.zeros_like(g).to(self.device) for g in self.grad_mean]

        for client in sampled_clients:
            client.grad_cal()
            client_gradients = client.client_gradients
            grad_sum = [g1 + g2 for g1, g2 in zip(grad_sum, client_gradients)]
        grad_sum = tuple(grad_sum)
        grad_mean_new = tuple(grad / len(sampled_clients) for grad in grad_sum)

        return tuple(0 * g1.detach() + (1 - 0) * g2 for g1, g2 in zip(self.grad_mean, grad_mean_new))

    def test(self, epoch):
        i = epoch + 1
        server_model = torch.jit.load("./models/central/server" + str(i) + ".pt", map_location=self.device)
        if self.model == 'ncf':
            simul_client_model = NeuralCollaborativeFiltering(self.ui_matrix.shape[0],self.ui_matrix.shape[1],self.latent_dim).to(self.device)
        elif self.model == 'svdpp':
            simul_client_model = SVDPP(self.ui_matrix.shape[0],self.ui_matrix.shape[1],self.latent_dim).to(self.device)
        elif self.model == 'deepfm':
            simul_client_model = DeepFM(self.ui_matrix.shape[0], self.ui_matrix.shape[1], self.latent_dim).to(self.device)
        else:
            raise ValueError('please choose a base model')
        simul_client_model.load_server_weights(server_model)

        hit_10_total = 0
        ndcg_10_total = 0
        hit_5_total = 0
        ndcg_5_total = 0
        for user in range(self.ui_matrix.shape[0]):
            user_ids=[user]
            current_ui_matrix = self.data_loader.get_ui_matrix(user_ids)
            loader = MatrixLoader(current_ui_matrix, dataloader=self.data_loader, user_ids=user_ids, thresh=args.thresh)
            test_batch = loader.get_test_batch()
            test_batch = test_batch.to(self.device)
            hr, ndcg = compute_metrics(model=simul_client_model, test_batch=test_batch, device=self.device)
            hr_5, ndcg_5 = compute_metrics_5(model=simul_client_model, test_batch=test_batch, device=self.device)
            hit_10_total += hr
            ndcg_10_total += ndcg
            hit_5_total += hr_5
            ndcg_5_total += ndcg_5

        ave_all = (hit_10_total + ndcg_10_total + hit_5_total + ndcg_5_total) / 4
        ave_10 = (hit_10_total + ndcg_10_total) / 2
        print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(hit_10_total, ndcg_10_total, hit_5_total,
                                                                      ndcg_5_total, ave_all, ave_10))
        self.metric_ndcg.append(ndcg_10_total)
        self.metric_hits.append(hit_10_total)
        self.metric_ndcg_5.append(ndcg_5_total)
        self.metric_hits_5.append(hit_5_total)
        logging.info(
            f"Test performance at the {epoch} rounds: N10: {ndcg_10_total:.2f}, H10: {hit_10_total:.2f}，N5: {ndcg_5_total:.2f}, H5: {hit_5_total:.2f}, all_ave: {ave_all}, all_10: {ave_10}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path of ratings.dat file')
    parser.add_argument('--num_clients', type=int, default=400, help='the numbers of clients')
    parser.add_argument('--model', type=str, default='ncf', help='the mode of model, you can choose from both ncf and gmf')
    parser.add_argument('--global_epochs', type=int, default=150, help='global epochs for aggregation')
    parser.add_argument('--local_epochs', type=int, default=5, help='global epochs for training of local epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learing rate')
    parser.add_argument('--filename', type=str, default='./ml-1m/ratings.dat')
    parser.add_argument('--npy_file', type=str, default='./ml-1m/ratings.npy')
    parser.add_argument('--thresh', type=float, default=0.1)
    parser.add_argument('--client_data_thresh', type=int, default=1)
    parser.add_argument('--lower_lamda', type=int, default=1)
    parser.add_argument('--higher_lamda', type=int, default=1)
    parser.add_argument('--mean_num', type=int, default=1)
    parser.add_argument('--aggregation', type=str, default='fedavg')
    args = parser.parse_args()

    if args.dataset == 'ml-100k':
        args.filename = './ml-100k/ratings.dat'
        args.npy_file = './ml-100k/ratings.npy'
    elif args.dataset == 'ml-1m':
        args.filename = './ml-1m/ratings.dat'
        args.npy_file = './ml-1m/ratings.npy'
    elif args.dataset == 'douban':
        args.filename = './douban/ratings.dat'
        args.npy_file = './douban/ratings.npy'
    else:
        raise ValueError('dataset is not found')

    logging.basicConfig(filename='metrics_info.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Drop program started with arguments: {}'.format(args))

    dataloader = MovielensDatasetLoader(filename=args.filename,
                                        npy_file=args.npy_file, thresh=args.thresh)
    seeds = {117623077}
    for s in seeds:
        fncf = FederatedNCF(
            data_loader=dataloader,
            num_clients=args.num_clients,
            user_per_client_range=[1,1],
            model=args.model,
            aggregation_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            batch_size=64,
            latent_dim=12,
            lr=args.lr,
            client_data_thresh=args.client_data_thresh,
            thresh=args.thresh,
            seed=s,
            lower_lamda=args.lower_lamda,
            higher_lamda=args.higher_lamda,
            mean_num=args.mean_num,
            aggregation=args.aggregation
        )
        fncf.train()
