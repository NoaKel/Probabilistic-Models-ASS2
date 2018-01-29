import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import operator
#from tabulate import tabulate


CLUSTER_NUM = 9
MIN_WORD_FREQ = 3
LAMBDA = 1.1
THRESHOLD = 25
K = 10
EPSILON = 0.0001

class article:
    def __init__(self, hist, topics, cluster):
        """
        article class
        :param hist: word histogram
        :param cluster: assigned cluster
        """
        self.hist = hist
        self.topics = topics
        self.cluster = cluster

class data:
    def __init__(self, file, topics_filename):
        """
        input data ctor
        :param file: input file
        """
        self.vocab = Counter
        self.file = []
        self.topics = []
        with open(file) as input_file:
            file_data = input_file.read()
        file_sen = file_data.splitlines()[1::2]
        file_topics = [line[1:-1].split('\t')[2:] for line in file_data.splitlines()[0::2]]

        # topics
        with open(topics_filename) as topics_file:
            topics_data = topics_file.read()
        self.topics_list = topics_data.split()
        self.topic_to_idx = {k: i for i, k in enumerate(self.topics_list)}
        self.idx_to_topic = {i: k for i, k in enumerate(self.topics_list)}

        # vocab hist
        words = ''.join(file_sen)[:-1]
        word_count = Counter(words.split(' '))
        self.vocab = Counter(el for el in word_count.elements() if word_count[el] > MIN_WORD_FREQ)

        # file class
        for sen, topics in zip(file_sen, file_topics):
            words = sen.split()
            self.file.append((Counter(words), topics))
            self.topics.append(topics)


class EM:
    def __init__(self, input):
        """
        EM class ctor
        :param input: input data
        """
        self.input = input
        self.articles = []

        self.init_val()
        self.iterate()

    def init_val(self):
        """
        init values uniform distribution
        :return: N/A
        """
        for i, file in enumerate(self.input.file):
            self.articles.append(article(file[0], file[1],self.input.idx_to_topic[i%9]))

    def e_init(self):
        """
        init e values for first iteration
        :return: N/A
        """
        self.w = {}
        for i in xrange(CLUSTER_NUM):
            for t in self.articles:
                if input.topic_to_idx[t.cluster] == i:
                    self.w[(i, t)] = 1.0
                else:
                    self.w[(i, t)] = 0.0

    def calc_z(self):
        """
        calc z for smoothing and m (maximal z)
        :return: N/A
        """
        self.z = {}
        self.m = {}

        # calc z_i_t
        for i in xrange(CLUSTER_NUM):
            for t in self.articles:
                self.z[(i, t)] = np.log(self.alpha[i]) + \
                (sum(np.log(self.p[(i, k)]) * appearances for k, appearances in t.hist.iteritems()))

        # calc m_t
        for t in self.articles:
            tmp_max = -float('inf')
            for i in xrange(CLUSTER_NUM):
                tmp_max = max(self.z[(i, t)] ,tmp_max)
            self.m[t] = tmp_max

    def e_stage(self):
        """
        E stage calc wit
        :return: N/A
        """
        self.w = {}

        # calc w_i_t
        for i in xrange(CLUSTER_NUM):
            for t in self.articles:
                if (self.z[(i, t)] - self.m[t]) < -1.0 * K:
                    self.w[(i, t)] = 0.0
                else:
                    nom = np.exp(self.z[(i,t)] - self.m[t])
                    denom = sum(np.exp(self.z[(j, t)] - self.m[t]) for j in xrange(CLUSTER_NUM) if (self.z[(j, t)] - self.m[t] >= -1.0 * K))
                    self.w[(i, t)] = float(nom)/denom

    def calc_alpha(self):
        """
        calc alpha, alpha smaller than epsilon will be rounded to epsilon
        :return: N/A
        """
        self.alpha = {}
        # calc alpha
        for i in xrange(CLUSTER_NUM):
            self.alpha[i] = sum(self.w[(i, t)] for t in self.articles)
            self.alpha[i] /= len(self.articles)
            if self.alpha[i] < EPSILON:
                self.alpha[i] = EPSILON
        # fix sum to 1
        sum_alphas = sum(self.alpha[i] for i in xrange(CLUSTER_NUM))
        for i in xrange(CLUSTER_NUM):
            self.alpha[i] /= sum_alphas

    def calc_p(self):
        """
        calc p with lidstone smoothing
        :return: N/A
        """
        self.p = {}
        nom = {}
        denom = {}
        for (i, t), prob in self.w.iteritems():
            for k in t.hist.iterkeys():
                if (i, k) not in nom or nom[(i, k)] is None:
                    nom[(i, k)] = 0.0
                nom[(i, k)] += t.hist[k] * prob
            if i not in denom or denom[i] is None:
                denom[i] = 0.0
            denom[i] += sum(t.hist.values()) * prob
        for (i, k), n in nom.iteritems():
            self.p[(i, k)] = float(n + LAMBDA) / (denom[i] + (len(self.input.vocab) * LAMBDA))

    def m_stage(self):
        """
        M stage calc alpha and p
        :return: N/A
        """
        self.calc_alpha()
        self.calc_p()

    def likelihood(self):
        """
        likelihood calc
        :return: likelihood
        """
        total_by_article = {}
        # Initiate
        for t in self.articles:
            total_by_article[t] = 0.0
        # Calculate for each article its likelihood
        for t in self.articles:
            exponent_sum = total_by_article[t]
            for i in xrange(CLUSTER_NUM):
                if self.z[(i, t)] - self.m[t] >= -1.0 * K:
                    exponent_sum += np.exp(self.z[(i,t)] - self.m[t])
            total_by_article[t] = np.log(exponent_sum) + self.m[t]
        # Sum all together
        return float(sum(total_by_article.itervalues()))

    def calc_perplexity(self, likelihood):
        """
        calc perplexity
        :param likelihood: likelihood
        :return: perplexity
        """
        total_words = sum(sum(t.hist.values()) for t in self.articles)
        return np.exp(-1 * (likelihood / float(total_words)))

    def calc_confusion_matrix(self):
        confusion = np.zeros((9, 9))
        for a in self.articles:
            articles_value = {key[0]: value for key, value in self.w.iteritems() if key[1] == a}
            max_cluster = max(articles_value.iteritems(), key=operator.itemgetter(1))[0]
            for t in a.topics:
                confusion[max_cluster][input.topic_to_idx[t]] += 1
        return confusion

    def calc_accuracy(self):
        confusion_matrix = self.calc_confusion_matrix()
        sum = 0.0
        for i in xrange(CLUSTER_NUM):
            sum += max(confusion_matrix[i])
        return sum / len(self.articles)

    def iterate(self):
        """
        EM main iterations
        :return: N/A
        """
        self.likelihoods = []
        self.perplexitties = []
        self.accuracies = []
        prev_likelihood = 0
        cur_likelihood = None
        while not cur_likelihood or (cur_likelihood - prev_likelihood > THRESHOLD):
            # E
            if not cur_likelihood:
                self.e_init()
            else:
                self.e_stage()

            # M
            self.m_stage()

            # likelihood
            self.calc_z()
            prev_likelihood = cur_likelihood if cur_likelihood else -float('inf')
            cur_likelihood = self.likelihood()

            # perplexity
            perplexity = self.calc_perplexity(cur_likelihood)

            # accuracy
            acc = self.calc_accuracy()

            # save for plots
            self.likelihoods.append(cur_likelihood)
            self.perplexitties.append(perplexity)
            self.accuracies.append(acc)
            print cur_likelihood, perplexity, acc

def plot_graph(y, name):
    """
    plots graphs
    :param y: y labels
    :param name: y axis name (will also be file name)
    :return: N/A
    """
    plt.figure()
    plt.plot(range(len(y)),y,linewidth=2.0)
    plt.xlabel("Iterations")
    plt.ylabel(name)
    plt.title(name + " vs Iterations")
    plt.savefig(name + ".png")



if __name__ == "__main__":
    development_set_filename = sys.argv[1]
    topics_filename = sys.argv[2]
    # input data
    input = data(development_set_filename, topics_filename)
    print 'vocab size', len(input.vocab)
    # EM algorithm
    em = EM(input)
    # graphs
    plot_graph(em.likelihoods, "lnLikelihood")
    plot_graph(em.perplexitties, "Perplexity")
    plot_graph(em.accuracies, "Accuracy")



