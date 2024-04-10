import datetime
import os

import numpy as np


class MovielensDatasetLoader:
    def __init__(self, filename='./ml-1m/ratings.dat',
                 npy_file='./ml-1m/ratings.npy',
                 num_movies=None,
                 num_users=None,
                 thresh=1):#thresh控制评分多少以上的数据是有效训练数据
        self.filename = filename
        self.npy_file = npy_file
        self.thresh = thresh
        self.rating_tuples, self.latest_ratings = self.read_ratings_file()
        if num_users is None:
            self.num_users = np.max(self.rating_tuples.T[0])  #应该是6040
        else:
            self.num_users = num_users
        if num_movies is None:
            self.num_movies = np.max(self.rating_tuples.T[1])
            # a = np.min(self.rating_tuples.T[1])
        else:
            self.num_movies = num_movies
        self.ratings = self.load_ui_matrix()

    def read_ratings(self):
        ratings = open(self.filename, 'r').readlines()
        data = np.array([[int(i) for i in rating[:-1].split("::")[:-1]] for rating in ratings])
        return data

    def read_ratings_file(self):
        latest_ratings = {}#表示该用户最后一次交互的物品，及时间
        data = []
        val_set = {}
        with open(self.filename, 'r') as f:
            for line in f:
                user_id, movie_id, rating, timestamp = map(int, line.strip().split("::"))
                timestamp = datetime.datetime.fromtimestamp(timestamp)
                data.append([user_id, movie_id, rating])  #
                user_id -= 1
                if user_id not in latest_ratings or timestamp > latest_ratings[user_id]['timestamp']: #and rating >= 3:
                    latest_ratings[user_id] = {"item_id": movie_id - 1,
                                               "timestamp": timestamp,
                                               "rating": rating
                                               }



        data = np.array(data)
        return data, latest_ratings #, val_set #前者用来形成ui矩阵成为训练集，中间作为测试集，后者作为验证集

    def generate_ui_matrix(self):
        data = np.zeros((self.num_users, self.num_movies))
        # TODO : get and remove the latest item from the train -> get the latest and remove it from train dataset
        # TODO : generate the testing iter : 99 -ve + 1 latest
        # TODO : evaluation metrics
        # TODO : add sigmoid and make the loss BCE
        for rating in self.rating_tuples: #将阈值设为self.thresh,大于此值时认为交互
            # data[rating[0] - 1][rating[1] - 1] = (rating[2] >= self.thresh).astype(np.int_) if self.thresh else rating[
            #     2]#让ui矩阵中只有0,1值的交互数据
            data[rating[0] - 1][rating[1] - 1] = rating[2] if self.thresh else rating[
                2]  # 让ui矩阵中带有评分值的交互数据
        return data

    def load_ui_matrix(self):
        if self.thresh:
            return self.generate_ui_matrix()
        if not os.path.exists(self.npy_file):
            ratings = self.generate_ui_matrix()
            np.save(self.npy_file, ratings)
        return np.load(self.npy_file)

    def get_ui_matrix(self, user_ids):
        return self.ratings[user_ids]

    def __str__(self) -> str:
        return f"MovieLens1M DataLoader\n" \
               f"Number of Users: {self.num_users}\n" \
               f"Number of Movies: {self.num_movies}\n" \
               f"UI_Matrix: {self.ratings.shape} "


if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    print(dataloader)
    print(dataloader.get_ui_matrix([0, 1, 2]))
    print(dataloader.ratings)
