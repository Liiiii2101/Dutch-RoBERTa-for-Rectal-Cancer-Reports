import torch
from lifelines.utils import concordance_index as con
from lifelines import KaplanMeierFitter, CoxPHFitter

def cox_loss(y_true, y_pred):
    time_value = torch.squeeze(y_true[0:, 0])
    event = torch.squeeze(y_true[0:, 1]).type(torch.bool)
    score = torch.squeeze(y_pred)

    ix = torch.where(event)[0]

    sel_time = time_value[ix]
    sel_mat = (sel_time.unsqueeze(1).expand(1, sel_time.size()[0],
                                            time_value.size()[0]).squeeze() <= time_value).float()

    p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score), axis=-1))

    loss = -torch.mean(p_lik)

    return loss


def concordance_index(y_true, y_pred, median_score=None):
    time_value = torch.squeeze(y_true[0:, 0]).cpu().numpy()
    event = torch.squeeze(y_true[0:, 1]).type(torch.bool).cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    ci  = con(time_value, y_pred, event)
    # if median_score:
    #     ci  = con(time_value, y_pred, event)
    #     median_score = np.median(y_pred)
    #     return ci, median_score
    # else:
    #     y_pred = [1 if i > median_score else 0 for i in y_pred]


    
    # time_1 = time_value.unsqueeze(1).expand(1, time_value.size()[0], time_value.size()[0]).squeeze()
    # event_1 = event.unsqueeze(1).expand(1, event.size()[0], event.size()[0]).squeeze()
    # ix = torch.where(torch.logical_and(time_1 < time_value, event_1))

    # s1 = y_pred[ix[0]]
    # s2 = y_pred[ix[1]]
    # ci = torch.mean((s1 < s2).float())

    return ci

