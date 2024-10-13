import math
import pickle


def save_sparse_matrix_to_pickle(sparse_matrix, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(sparse_matrix, f)


def fast_evaluation(epoch, measure, bestPerformance, logger):
    print('Evaluating the model...')

    if len(bestPerformance) > 0:
        count = 0
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        for k in bestPerformance[1]:
            if bestPerformance[1][k] > performance[k]:
                count += 1
            else:
                count -= 1
        if count < 0:
            bestPerformance[1] = performance
            bestPerformance[0] = epoch + 1
            # bestPerformance[2] = rec_list
    else:
        bestPerformance.append(epoch + 1)
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        bestPerformance.append(performance)

    print('-' * 120)
    print('Real-Time Ranking Performance ' + ' (Top-' + str(20) + ' Item Recommendation)')
    measure = [m.strip() for m in measure[1:]]
    metrics_dict = {item.split(':')[0]: float(item.split(':')[1]) for item in measure}

    logger.info(f'epoch {epoch + 1} tested, Recall@20: {metrics_dict["Recall"]:.5f}, '
                f'NDCG@20: {metrics_dict["NDCG"]:.5f}')
    print('*Current Performance*')
    print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
    bp = ''
    # for k in self.bestPerformance[1]:
    #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
    bp += 'Hit Ratio' + ':' + str(bestPerformance[1]['Hit Ratio']) + '  |  '
    bp += 'Precision' + ':' + str(bestPerformance[1]['Precision']) + '  |  '
    bp += 'Recall' + ':' + str(bestPerformance[1]['Recall']) + '  |  '
    # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
    bp += 'NDCG' + ':' + str(bestPerformance[1]['NDCG'])

    print('*Best Performance* ')
    print('Epoch:', str(bestPerformance[0]) + ',', bp)
    print('-' * 120)

    return bestPerformance


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num / total_num, 5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall), 5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return round(error / count, 5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return round(math.sqrt(error / count), 5)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res), 5)


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        # MAP = Measure.MAP(origin, predicted, n)
        # indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure
