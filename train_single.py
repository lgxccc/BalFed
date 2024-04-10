import copy
import math
import numpy as np
import torch
from dataloader import MovielensDatasetLoader
from ncf_model import NeuralCollaborativeFiltering
from svdpp_model import SVDPP
from gmf_model import GMF
from math import cos, pi
import torch.autograd as autograd
from deepFM import DeepFM

def setdiff2d_set(arr1, arr2):
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return np.array(list(set1 - set2))

class MatrixLoader:
    def __init__(self,
                 ui_matrix,
                 dataloader: MovielensDatasetLoader = None,
                 default=None,
                 user_ids=None,
                 thresh=1):
        self.ui_matrix = ui_matrix
        self.positives = np.argwhere(self.ui_matrix >= thresh)
        self.negatives = np.argwhere(self.ui_matrix == 0)
        for i, usr_id in enumerate(user_ids):
            self.positives[self.positives[:, 0] == i, 0] = usr_id
            self.negatives[self.negatives[:, 0] == i, 0] = usr_id
        self.relabel = False
        self.user_ids = user_ids
        self.dataloader = dataloader
        if user_ids:
            test_interactions = np.array(
                [[usr_id, dataloader.latest_ratings[usr_id]["item_id"]] for usr_id in user_ids])
            mask = np.array([not np.array_equal(row, rows_to_remove_i) for row in self.positives for rows_to_remove_i in
                             test_interactions]).reshape(self.positives.shape[0], test_interactions.shape[0]).all(axis=1)
            self.positives = self.positives[mask]
        if default is None:
            self.default = np.array([[0, 0]]), np.array([0])
        else:
            self.default = default

    def delete_indexes(self, indexes, arr="pos"):
        if arr == "pos":
            self.positives = np.delete(self.positives, indexes, 0)
        else:
            self.negatives = np.delete(self.negatives, indexes, 0)

    def get_batch(self, batch_size):
        if self.positives.shape[0] < batch_size // 4 or self.negatives.shape[0] < batch_size - batch_size // 4:
            return torch.tensor(self.default[0]), torch.tensor(self.default[1])
        try:
            pos_indexes = np.random.choice(self.positives.shape[0], batch_size // 4, replace=False)
            neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size // 4, replace=False)
            pos = self.positives[pos_indexes]
            neg = self.negatives[neg_indexes]
            self.delete_indexes(pos_indexes, "pos")
            self.delete_indexes(neg_indexes, "neg")
            batch = np.concatenate((pos, neg), axis=0)
            if batch.shape[0] != batch_size:
                return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
            np.random.shuffle(batch)
            y = np.array([self.dataloader.ratings[i][j] for i, j in batch])
            return batch, y
        except Exception as exp:
            print(exp)
            return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()

    def get_test_batch(self):
        user_id = np.random.choice(self.user_ids)
        pos = np.array([user_id, self.dataloader.latest_ratings[user_id]["item_id"]])
        neg_indexes = np.random.choice(self.negatives.shape[0], 99)
        neg = self.negatives[neg_indexes]
        batch = np.concatenate((pos.reshape(1, -1), neg), axis=0)
        return torch.tensor(batch)


class NCFTrainer:
    def __init__(self,
                 data_loader: MovielensDatasetLoader,
                 user_ids,
                 epochs,
                 batch_size,
                 model,
                 thresh,
                 lower_thresh,
                 upper_thresh,
                 mean,
                 mean_num,
                 latent_dim=32,
                 device=None):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.loader = None
        self.thresh = thresh
        self.user_ids = user_ids
        self.relabel = False
        self.unlabeled = np.zeros((0, 2))
        self.data_loader = data_loader
        self.ui_matrix = self.data_loader.get_ui_matrix(self.user_ids)
        self.nonzero = np.sum(self.ui_matrix > 0)
        self.local_pos_prob_list = []
        self.local_pos_prob = 1
        self.local_thresh = 1
        self.global_thresh = 1
        self.mean = mean
        self.initialize_loader()
        if self.nonzero < lower_thresh: #randomly sample q unlabeled elements from items that the user has not interacted with
            zero_indices = np.where(self.ui_matrix == 0)
            random_zero_indices = np.random.choice(zero_indices[1], size=mean_num*round(mean), replace=False)
            self.unlabeled = np.column_stack((np.full(random_zero_indices.shape, self.user_ids), random_zero_indices))
            self.ui_matrix[0, random_zero_indices] = -1
            self.unlabeled_indice = random_zero_indices
            self.relabel = True

        if self.nonzero > upper_thresh: #randomly dowmsampling
            nozero_indices = np.where(self.ui_matrix > 0)
            zero_num = len(nozero_indices[1]) - upper_thresh
            random_zero_indices = np.random.choice(nozero_indices[1], size=zero_num, replace=False)
            self.ui_matrix[0, random_zero_indices] = -2
        self.nonzero = np.sum(self.ui_matrix > 0) + np.sum(self.ui_matrix < -1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initialize_loader()
        if model == 'ncf':
            self.model = NeuralCollaborativeFiltering(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                self.latent_dim).to(self.device)
            self.sever_model = NeuralCollaborativeFiltering(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                       self.latent_dim).to(self.device)
        elif model == 'svdpp':
            self.model = SVDPP(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                      self.latent_dim).to(self.device)
            self.sever_model = SVDPP(self.data_loader.ratings.shape[0],self.data_loader.ratings.shape[1],
                                                            self.latent_dim).to(self.device)
        elif model == 'deepfm':
            self.model = DeepFM(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                               self.latent_dim).to(self.device)
            self.sever_model = DeepFM(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                     self.latent_dim).to(self.device)
        else:
            raise ValueError('please specify base model')
        self.client_gradients = [torch.zeros_like(p) for p in self.model.parameters()]

    def initialize_loader(self):
        self.loader = MatrixLoader(self.ui_matrix, dataloader=self.data_loader, user_ids=self.user_ids, thresh=self.thresh)

    def train_batch(self, x, y, optimizer, grad_mean, grad_update_times):
        self.model.train()  # set the model to training mode
        optimizer.zero_grad()  # zero the gradients
        y_ = self.model(x)  # forward pass
        loss_erm = torch.nn.functional.binary_cross_entropy(y_, y)
        grad_client = autograd.grad(loss_erm, self.model.parameters(), create_graph=True, allow_unused=True, retain_graph=True)

        penalty_value = 0
        for g_client, g_mean in zip(grad_client, grad_mean):
            if g_client is not None:
                penalty_value += (g_client - g_mean).pow(2).sum()
        penalty_weight_base = 0.01
        penalty_weight = penalty_weight_base * math.exp(-(self.local_pos_prob/self.global_thresh -1))
        # penalty_weight = penalty_weight_base
        loss_all = loss_erm + penalty_weight * penalty_value
        # loss_all = loss_erm
        loss_all.backward()
        optimizer.step()
        grad_update_times += 1
        # optimizer.zero_grad()
        return loss_erm.item(), y_.detach(), grad_update_times

    def train_model(self, optimizer, epoch_global, grad_mean,epochs_local=None):
        self.initialize_loader()
        epoch = 0
        grad_update_times = 0
        server_model = torch.jit.load("./models/central/server" + str(epoch_global) + ".pt", map_location=self.device)
        self.sever_model.load_server_weights(server_model)
        progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": []}
        running_loss, running_hr, running_ndcg = 0, 0, 0
        prev_running_loss, prev_running_hr, prev_running_ndcg = 0, 0, 0
        if epochs_local is None:
            epochs_local = self.epochs
        steps, prev_steps, prev_epoch, count, step_total = 0, 0, 0, 0, 0
        while epoch < epochs_local:
            x, y = self.loader.get_batch(self.batch_size)
            if x.shape[0] < self.batch_size:
                prev_running_loss, prev_running_hr, prev_running_ndcg = running_loss, running_hr, running_ndcg
                running_loss = 0
                running_hr = 0
                running_ndcg = 0
                prev_steps = steps
                step_total += steps
                epoch += 1
                self.initialize_loader()
                x, y = self.loader.get_batch(self.batch_size)
            y[y > 0.1] = 1
            y[y <= 0.1] = 0
            x = torch.tensor(x).int()
            y = torch.tensor(y).float()
            x, y = x.to(self.device), y.to(self.device)
            loss, y_, grad_update_times= self.train_batch(x, y, optimizer, grad_mean, grad_update_times)
            running_loss += loss
            if epoch != 0 and steps == 0:
                results = {"epoch": prev_epoch, "loss": prev_running_loss / (prev_steps + 1)}
            else:
                results = {"epoch": prev_epoch, "loss": running_loss / (steps + 1)}
            if prev_epoch != epoch:
                progress["epoch"].append(results["epoch"])
                progress["loss"].append(results["loss"])
                prev_epoch += 1

            if epoch == 0:
                positive_indices = np.where(y.cpu().numpy() == 1)
                pos_pred_prob = y_[positive_indices]
                self.local_pos_prob_list.append(np.mean(pos_pred_prob.cpu().numpy()))
        self.local_pos_prob = sum(self.local_pos_prob_list)/(len(self.local_pos_prob_list)+0.000000001)
        self.local_pos_prob_list = []


        if epoch_global > 1 and self.relabel: #perform data augmentation
            unlabel_train = self.unlabeled
            self.initialize_loader()
            if len(unlabel_train) > 0:
                unlabel_train = torch.tensor(unlabel_train).int().to(self.device)
                with torch.no_grad():
                    client_pred_unlabel_train = self.model(unlabel_train)
                    server_pred_unlabel_train = self.sever_model(unlabel_train)
                    client_pred_train = client_pred_unlabel_train.to('cpu').numpy()
                    server_pred_train = server_pred_unlabel_train.to('cpu').numpy()
                    clean_confid_idx = np.where((client_pred_train > self.local_thresh) & (server_pred_train > self.global_thresh))[0]
                    unlabel_train = unlabel_train.to('cpu').numpy()
                    if len(clean_confid_idx) > 0:
                        clean_confid_data = unlabel_train[clean_confid_idx]
                        for item in clean_confid_data:
                            row_idx, col_idx = item
                            self.ui_matrix[0, col_idx] = 6
                            self.loader.dataloader.ratings[
                                self.user_ids[0], col_idx] = 6
                        self.unlabeled = setdiff2d_set(self.unlabeled, clean_confid_data)
                        self.initialize_loader()
                        self.nonzero = np.sum(self.ui_matrix > 0)

        self.initialize_loader()
        r_results = {"num_users": self.ui_matrix.shape[0]}
        r_results.update({i: results[i] for i in ["loss"]})
        return r_results, progress

    def grad_cal(self):
        epoch = 0
        if isinstance(self.model, NeuralCollaborativeFiltering):
            self.model.join_output_weights()
        grad_accumulator = [torch.zeros_like(p) for p in self.model.parameters()]
        grad_update_times = 0
        self.simul_client_model = copy.deepcopy(self.model)

        while epoch < 1:
            x, y = self.loader.get_batch(self.batch_size)
            if x.shape[0] < self.batch_size:
                epoch += 1
                self.initialize_loader()
                x, y = self.loader.get_batch(self.batch_size)
            y[y > 0.1] = 1
            y[y <= 0.1] = 0
            x = torch.tensor(x).int()
            y = torch.tensor(y).float()
            x, y = x.to(self.device), y.to(self.device)
            y_ = self.simul_client_model(x)
            loss = torch.nn.functional.binary_cross_entropy(y_, y)
            grad_client = autograd.grad(loss, self.simul_client_model.parameters(), create_graph=False, allow_unused=True)
            for grad, accumulator in zip(grad_client, grad_accumulator):
                if grad is not None:
                    accumulator += grad
            grad_update_times += 1
        self.client_gradients = [accumulator / grad_update_times for accumulator in grad_accumulator]

    def train(self, ncf_optimizer, epoch, user_dict, grad_mean, return_progress=False):
        if isinstance(self.model, NeuralCollaborativeFiltering):
            self.model.join_output_weights()
        results, progress = self.train_model(ncf_optimizer, epoch, grad_mean)
        if return_progress:
            return results, progress,user_dict
        else:
            return results,user_dict


