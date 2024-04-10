import numpy as np
import torch



def hit(ng_item, pred_items):
    if ng_item in pred_items:
        return 1
    return 0


def ndcg(ng_item, pred_items):
    if ng_item in pred_items:
        index = pred_items.index(ng_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def compute_metrics(model,
                    test_batch,
                    device,
                    top_k=10):
    model.eval()
    predictions = model(test_batch)
    _, indices = torch.topk(predictions, top_k)
    recommends = torch.take(test_batch[:, 1], indices).cpu().numpy().tolist()
    pos = test_batch[0, 1]
    return hit(pos, recommends), ndcg(pos, recommends)

def compute_metrics_5(model,
                    test_batch,
                    device,
                    top_k=5):
    model.eval()
    predictions = model(test_batch)
    _, indices = torch.topk(predictions, top_k)
    recommends = torch.take(test_batch[:, 1], indices).cpu().numpy().tolist()
    pos = test_batch[0, 1]
    return hit(pos, recommends), ndcg(pos, recommends)