import copy
import random
import time
import numpy as np
from cec2013.cec2013 import *

nofo = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]


class WSNADE(object):
    def __init__(self, Np, dim, f):
        self.Np = Np
        self.dim = dim
        self.FEs = 0
        self.F9 = 0.9
        self.F5 = 0.5
        self.CR = 0.1
        self.T = int(f.get_maxfes() / self.Np / 8)              # 停滞代数阈值
        self.archiveT = 2*self.T                                # 直接存入archive的阈值
        self.Population = np.zeros((self.Np, self.dim))         # 种群
        self.Fitness = np.zeros(self.Np)                        # 适应值
        self.u = np.zeros(self.dim)                             # 子代
        self.archive = np.zeros((0, self.dim))                  # 定义的外部存档
        self.archiveFitness = np.zeros(0)                       # 外部存档个体的适应值
        self.stagnation = np.zeros(self.Np)                     # 每个个体的停滞代数

        self.gMaxFit = -np.inf                                  # 全局最大适应值
        self.EG = np.zeros((self.Np, self.Np))                  # 距离矩阵

        self.prohibition = np.zeros(self.Np)
        self.isPerfectIndividual = np.zeros(self.Np)            # 为1则为完美个体，不操作不评估

        self.eliteSet = []                                      # 精英集合

        self.nSize = 10
        self.maxSize = 20
        self.minSize = 3

    # 初始化
    def initialize(self, f):
        for i in range(self.Np):
            for j in range(self.dim):
                self.Population[i][j] = random.uniform(f.get_lbound(j), f.get_ubound(j))
            self.Fitness[i] = f.evaluate(self.Population[i])
            self.FEs += 1
            if self.Fitness[i] > self.gMaxFit:
                self.gMaxFit = self.Fitness[i]

    # 最近niSize个个体的距离平均值的一半
    def getR(self, index):
        dist = np.zeros(self.Np)
        for i in range(self.Np):
            dist[i] = self.EG[index][i]

        dist.sort()
        sum = 0
        for i in range(self.nSize):
            sum += dist[i+1]

        return sum/self.nSize/2

    def do(self, f):
        MaxFEs = f.get_maxfes()
        recordFEs = int(MaxFEs/4)
        while self.FEs < MaxFEs:
            # 更新个体间的距离
            for i in range(self.Np):
                for j in range(i + 1, self.Np):
                    temp = 0
                    for k in range(self.dim):
                        temp += (self.Population[i][k] - self.Population[j][k]) ** 2
                    self.EG[i][j] = np.sqrt(temp)
                    self.EG[j][i] = self.EG[i][j]

            # 对每个个体进行操作
            for i in range(self.Np):
                if self.isPerfectIndividual[i] == 0:

                    # 1.WANT
                    R = self.getR(i)
                    niche = [i]
                    left = 0
                    while True:
                        count = 0
                        for j in range(left, len(niche)):
                            for k in range(self.Np):
                                if self.EG[niche[j]][k] <= 2*R:
                                    niche.append(k)
                                    count += 1
                            left += 1
                        niche = list(set(niche))
                        if (count == 0 and len(niche) >= self.minSize) or len(niche) >= self.maxSize:
                            break

                        if len(niche) < self.minSize:
                            R *= 1.1
                            left = 0

                    # 2.PDM Strategy
                    rank = 0
                    for j in range(len(niche)):
                        if self.Fitness[i] > self.Fitness[niche[j]]:
                            rank += 1

                    pr = rank / len(niche)

                    # 2.1 small-scale mutation strategy
                    if random.uniform(0, 1) < pr:
                        r1, r2 = 0, 0
                        while r1 == r2 or r1 == 0 or r2 == 0:
                            r1 = random.randint(0, len(niche) - 1)
                            r2 = random.randint(0, len(niche) - 1)
                        self.u = self.Population[i] + self.F5 * (self.Population[niche[r1]] - self.Population[niche[r2]])
                    # 2.2 large-scale mutation strategy
                    else:
                        if random.uniform(0, 1) < 0.5 and self.gMaxFit - self.Fitness[i] > 0.1 and self.prohibition[i] == 0:
                            r1, r2, r3 = 0, 0, 0
                            while r1 == r2 or r2 == r3 or r1 == r3:
                                r1 = random.randint(0, len(niche) - 1)
                                r2 = random.randint(0, len(niche) - 1)
                                r3 = random.randint(0, len(niche) - 1)
                            self.u = self.Population[niche[r1]] + self.F9 * (
                                        self.Population[niche[r2]] - self.Population[niche[r3]])
                        else:
                            r1, r2 = 0, 0
                            while r1 == r2 or r1 == 0 or r2 == 0:
                                r1 = random.randint(0, len(niche) - 1)
                                r2 = random.randint(0, len(niche) - 1)
                            self.u = self.Population[i] + self.F9 * (
                                    self.Population[niche[r1]] - self.Population[niche[r2]])

                    # 越界处理
                    for j in range(self.dim):
                        if self.u[j] > f.get_ubound(j):
                            self.u[j] = f.get_ubound(j)
                        elif self.u[j] < f.get_lbound(j):
                            self.u[j] = f.get_lbound(j)

                    # 交叉
                    j_rand = random.randint(0, self.dim - 1)
                    for j in range(self.dim):
                        p = random.uniform(0, 1)
                        if not (p <= self.CR or j == j_rand):
                            self.u[j] = self.Population[i][j]

                    # 3.MLR Strategy
                    fit = self.Fitness[i]
                    flagI = True

                    # 3.1 Exclusion Mechanism
                    if len(self.eliteSet) > 0:
                        dist = np.zeros(len(self.eliteSet))
                        for j in range(len(self.eliteSet)):
                            temp = 0
                            for k in range(self.dim):
                                temp += (self.u[k] - self.eliteSet[j][k]) ** 2
                            dist[j] = np.sqrt(temp)

                        for j in range(len(dist)):
                            if dist[j] <= 0.1:
                                flagI = False
                                break

                        # 生成的个体在已存在的峰值上，不评估该子代且重置该父代
                        if not flagI:
                            # 重置自己
                            for j in range(self.dim):
                                self.Population[i][j] = random.uniform(f.get_lbound(j), f.get_ubound(j))
                            self.Fitness[i] = f.evaluate(self.Population[i])
                            self.FEs += 1
                            if self.Fitness[i] > self.gMaxFit:
                                self.gMaxFit = self.Fitness[i]
                            self.stagnation[i] = 0
                            self.prohibition[i] = 1

                            for j in range(self.Np):
                                temp = 0
                                for k in range(self.dim):
                                    temp += (self.Population[i][k] - self.Population[j][k]) ** 2
                                self.EG[i][j] = np.sqrt(temp)
                                self.EG[j][i] = self.EG[i][j]
                        else:
                            fit = f.evaluate(self.u)
                            self.FEs += 1
                    else:
                        fit = f.evaluate(self.u)
                        self.FEs += 1

                    # 3.2 Stagnation Mechanism
                    if flagI:
                        if fit > self.gMaxFit:
                            self.gMaxFit = fit
                        # 选择
                        if fit > self.Fitness[i]:
                            self.Fitness[i] = fit
                            self.Population[i] = self.u
                            self.stagnation[i] = 0

                            for j in range(self.Np):
                                temp = 0
                                for k in range(self.dim):
                                    temp += (self.Population[i][k] - self.Population[j][k]) ** 2
                                self.EG[i][j] = np.sqrt(temp)
                                self.EG[j][i] = self.EG[i][j]

                        else:
                            self.stagnation[i] += 1
                            self.prohibition[i] = 0
                            if self.FEs >= recordFEs:
                                if self.stagnation[i] >= self.T:
                                    if self.gMaxFit - self.Fitness[i] > 0.1:
                                        # 大于停滞阈值，且不好，判断是否为niche最好个体，是则重置全部
                                        p = 0
                                        for j in niche:
                                            if self.Fitness[i] >= self.Fitness[j]:
                                                p += 1
                                        if p == len(niche):
                                            # 重置此小生境内全部个体
                                            for j in range(len(niche)):
                                                for k in range(self.dim):
                                                    self.Population[niche[j]][k] = random.uniform(f.get_lbound(k),
                                                                                                  f.get_ubound(k))
                                                self.Fitness[niche[j]] = f.evaluate(self.Population[niche[j]])
                                                self.FEs += 1
                                                if self.Fitness[niche[j]] > self.gMaxFit:
                                                    self.gMaxFit = self.Fitness[niche[j]]
                                                self.stagnation[niche[j]] = 0
                                                self.prohibition[niche[j]] = 1

                                                for m in range(self.Np):
                                                    temp = 0
                                                    for k in range(self.dim):
                                                        temp += (self.Population[niche[j]][k] - self.Population[m][k]) ** 2
                                                    self.EG[niche[j]][m] = np.sqrt(temp)
                                                    self.EG[m][niche[j]] = self.EG[niche[j]][m]
                                        else:
                                            # 否则只重置自己
                                            for j in range(self.dim):
                                                self.Population[i][j] = random.uniform(f.get_lbound(j), f.get_ubound(j))
                                            self.Fitness[i] = f.evaluate(self.Population[i])
                                            self.FEs += 1
                                            if self.Fitness[i] > self.gMaxFit:
                                                self.gMaxFit = self.Fitness[i]
                                            self.stagnation[i] = 0
                                            self.prohibition[i] = 1

                                            for j in range(self.Np):
                                                temp = 0
                                                for k in range(self.dim):
                                                    temp += (self.Population[i][k] - self.Population[j][k]) ** 2
                                                self.EG[i][j] = np.sqrt(temp)
                                                self.EG[j][i] = self.EG[i][j]

                                    elif self.gMaxFit - self.Fitness[i] <= 0.00001:
                                        # 如果此小生境极好了，直接存入存档
                                        self.archive = np.append(arr=self.archive, values=[self.Population[i]], axis=0)
                                        self.archiveFitness = np.append(arr=self.archiveFitness, values=[self.Fitness[i]], axis=0)

                                        # 重置此小生境中除xi外其他全部个体
                                        for j in range(len(niche)):
                                            if niche[j] != i:
                                                for k in range(self.dim):
                                                    self.Population[niche[j]][k] = random.uniform(f.get_lbound(k),
                                                                                                  f.get_ubound(k))
                                                self.Fitness[niche[j]] = f.evaluate(self.Population[niche[j]])
                                                self.FEs += 1
                                                if self.Fitness[niche[j]] > self.gMaxFit:
                                                    self.gMaxFit = self.Fitness[niche[j]]
                                                self.stagnation[niche[j]] = 0
                                                self.prohibition[niche[j]] = 1
                                                self.isPerfectIndividual[niche[j]] = 0

                                                for m in range(self.Np):
                                                    temp = 0
                                                    for k in range(self.dim):
                                                        temp += (self.Population[niche[j]][k] - self.Population[m][k]) ** 2
                                                    self.EG[niche[j]][m] = np.sqrt(temp)
                                                    self.EG[m][niche[j]] = self.EG[niche[j]][m]

                                        self.isPerfectIndividual[i] = 1
                                        self.eliteSet.append(self.Population[i])

                                    elif self.stagnation[i] >= self.archiveT:
                                        self.archive = np.append(arr=self.archive, values=[self.Population[i]], axis=0)
                                        self.archiveFitness = np.append(arr=self.archiveFitness, values=[self.Fitness[i]], axis=0)
                                        # 只重置自己
                                        for j in range(self.dim):
                                            self.Population[i][j] = random.uniform(f.get_lbound(j), f.get_ubound(j))
                                        self.Fitness[i] = f.evaluate(self.Population[i])
                                        self.FEs += 1
                                        if self.Fitness[i] > self.gMaxFit:
                                            self.gMaxFit = self.Fitness[i]
                                        self.stagnation[i] = 0
                                        self.prohibition[i] = 1

                                        for j in range(self.Np):
                                            temp = 0
                                            for k in range(self.dim):
                                                temp += (self.Population[i][k] - self.Population[j][k]) ** 2
                                            self.EG[i][j] = np.sqrt(temp)
                                            self.EG[j][i] = self.EG[i][j]

            # print(f'FEs=={self.FEs}')


def getNP(func):
    if 1 <= func <= 5:
        return 100
    elif 7 <= func <= 9:
        return 600
    elif func == 6 or func == 10 or func == 20:
        return 200
    else:
        return 400

def main():
    func = 20
    record, record1, record2, record3, record4, record5 = [], [], [], [], [], []
    for i in range(51):
        f = CEC2013(func)
        Np = getNP(func)
        wsnade = WSNADE(Np=Np, dim=f.get_dimension(), f=f)

        begin = time.time()
        wsnade.initialize(f)
        wsnade.do(f)

        population = np.append(arr=wsnade.Population, values=wsnade.archive, axis=0)
        # # 输出
        accuracy1, accuracy2, accuracy3, accuracy4, accuracy5 = 0.1, 0.01, 0.001, 0.0001, 0.00001
        count1, seeds1 = how_many_goptima(population, f, accuracy1)
        count2, seeds2 = how_many_goptima(population, f, accuracy2)
        count3, seeds3 = how_many_goptima(population, f, accuracy3)
        count4, seeds4 = how_many_goptima(population, f, accuracy4)
        count5, seeds5 = how_many_goptima(population, f, accuracy5)
        record1.append(count1), record2.append(count2), record3.append(count3), record4.append(count4), record5.append(
            count5)
        end = time.time()
        # print(f'外部存档大小为{len(myde.archive)}')
        # print(f'In the accuracy == 0.1, there exist {count1} global optimizers.')
        # print(f'In the accuracy == 0.01, there exist {count2} global optimizers.')
        # print(f'In the accuracy == 0.001, there exist {count3} global optimizers.')
        # print(f'In the accuracy == 0.0001, there exist {count4} global optimizers.')
        # print(f'In the accuracy == 0.00001, there exist {count5} global optimizers.')
        # print(f'Global optimizers: {seeds4}')
        # print(f'所用时间为：{end - begin}\n')

        # 记录到文件中
        save = open(f"WSNADE/f{func}.txt", "a", encoding="utf-8")
        save.write(f'====================================          {i + 1}\n')
        save.write(f'外部存档大小为{len(wsnade.archive)}\n')
        save.write(f'In the accuracy == 0.1, there exist {count1} global optimizers.\n')
        save.write(f'In the accuracy == 0.01, there exist {count2} global optimizers.\n')
        save.write(f'In the accuracy == 0.001, there exist {count3} global optimizers.\n')
        save.write(f'In the accuracy == 0.0001, there exist {count4} global optimizers.\n')
        save.write(f'In the accuracy == 0.00001, there exist {count5} global optimizers.\n')
        save.write(f"第{i + 1}次所用时间为：{end - begin}\n")
        save.write("\n\n")
        save.close()

    record.append(record1), record.append(record2), record.append(record3), record.append(record4), record.append(
        record5)
    # PR, SR
    PR, SR = [], []
    for i in range(len(record)):
        PR.append(np.mean(record[i]) / nofo[func - 1])
        count = 0
        for j in range(len(record[i])):
            if record[i][j] == nofo[func - 1]:
                count += 1
        SR.append(count / len(record[i]))

    save = open(f"WSNADE/f{func}.txt", "a", encoding="utf-8")
    save.write(f'{len(record1)}次的峰数分别为：\n')
    save.write(f'In the accuracy == 0.1:\nrecord == {record[0]}\nPR is : {PR[0]}\nSR is : {SR[0]}\n\n')
    save.write(f'In the accuracy == 0.01:\nrecord == {record[1]}\nPR is : {PR[1]}\nSR is : {SR[1]}\n\n')
    save.write(f'In the accuracy == 0.001:\nrecord == {record[2]}\nPR is : {PR[2]}\nSR is : {SR[2]}\n\n')
    save.write(f'In the accuracy == 0.0001:\nrecord == {record[3]}\nPR is : {PR[3]}\nSR is : {SR[3]}\n\n')
    save.write(f'In the accuracy == 0.00001:\nrecord == {record[4]}\nPR is : {PR[4]}\nSR is : {SR[4]}\n\n')
    save.close()


if __name__ == '__main__':
    main()
