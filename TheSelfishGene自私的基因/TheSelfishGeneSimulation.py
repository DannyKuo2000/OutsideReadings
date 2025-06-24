# This a simulation program from "The Selfish Gene" Chap6
# 模擬了四種爭搶地盤的行為模式： 0.鷹 1.鴿子 2.復仇者 3.欺負弱小者
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# ====== 個體 ======
class Agent:
    def __init__(self, strategy_index):
        self.strategy = strategy_index
        self.fitness = 0

# ====== 模擬模型 ======
class EvolutionSimulator:
    def __init__(self, payoff_matrix, mutation_matrix, interaction_random_matrix,
                 population_init, population_cap, logistic_r, attack_rate, handling_time, pairing_num, generations, fix_pairing_num):
        self.payoff_matrix = payoff_matrix  # shape (n, n, 2)
        self.mutation_matrix = mutation_matrix  # shape (n, n)
        self.interaction_random_matrix = interaction_random_matrix
        self.population_now = population_init
        self.population_cap = population_cap
        self.logistic_r = logistic_r
        self.attack_rate = attack_rate
        self.handling_time = handling_time
        self.pairing_num = pairing_num
        self.generations = generations
        self.fix_pairing_num = fix_pairing_num

        self.num_strategies = payoff_matrix.shape[0]
        self.population = [] # 所有生物的list
        self.history_ratio = []
        self.history_num = []

    def initialize_population(self, initial_distribution):
        self.population = []
        counts = (initial_distribution * self.population_now).astype(int)

        # 為了避免總和不精確造成個體數不足，多出的人補給比例最高的策略
        remainder = self.population_now - np.sum(counts)
        if remainder > 0:
            max_idx = np.argmax(initial_distribution)
            counts[max_idx] += remainder

        # 根據比例創建個體
        for strategy_index, count in enumerate(counts):
            self.population.extend([Agent(strategy_index) for _ in range(count)])
        random.shuffle(self.population)
    
    def holling_type_II_interactions(self, population_now, attack_rate=0.0001, handling_time=0.01):  # Holling type II model(每個物種時間有限): N:互動對象數量, a:搜尋效率(attack rate), h:處理時間(handling time)
        return int((attack_rate * population_now**2) / (1 + handling_time * population_now))

    def logistic_population_change(self, population_now, population_cap, logistic_r):  # logistic model
        return int(population_now + logistic_r * population_now * (1 - population_now / population_cap))

    def run(self):
        for gen in trange(self.generations, desc="Simulating Generations"):
            for agent in self.population:
                agent.fitness = 0

            # 進行互動(隨機配對)
            if self.fix_pairing_num == False:
                self.pairing_num = self.holling_type_II_interactions(self.population_now, self.attack_rate, self.handling_time)

            for _ in range(self.population_now * self.pairing_num):
                a, b = random.sample(self.population, 2)
                i, j = a.strategy, b.strategy

                # 加入爭奪輸贏的隨機性(後面可以用戰力指數取代)
                if self.interaction_random_matrix[i][j] == 1:
                    if random.random() < 0.5:
                        reward_a, reward_b = self.payoff_matrix[i][j]
                    else:
                        reward_b, reward_a = self.payoff_matrix[i][j]
                else:
                    reward_a, reward_b = self.payoff_matrix[i][j]

                a.fitness += reward_a
                b.fitness += reward_b

            # 適應度選擇 + 突變
            fitnesses = np.array([agent.fitness for agent in self.population])
            strategies = np.array([agent.strategy for agent in self.population])
            min_fit = fitnesses.min() 
            if min_fit < 0:
                fitnesses = fitnesses - min_fit  # shift to make min = 0(避免負分難以計算抽樣)

            # === 分布視覺化（每幾代畫一次） ===
            if gen % 5  == 0 or gen == self.generations - 1:
                self.plot_fitness_distribution(fitnesses, strategies, gen)
            
            total_fitness = fitnesses.sum()
            if total_fitness == 0:
                # 所有 fitness 一樣（或都是負的），隨機選擇
                probs = None
            else:
                probs = fitnesses / total_fitness

            # Calculation the next generation population
            self.population_now = self.logistic_population_change(self.population_now, self.population_cap, self.logistic_r) # logistic model
            # 加權隨機抽樣,選出下一代的種類: k: 抽出的個數
            selected = random.choices(self.population, weights=probs, k=self.population_now) 

            new_population = []
            for agent in selected:
                i = agent.strategy
                new_strategy = np.random.choice(self.num_strategies, p=self.mutation_matrix[i])  # 根據前面的抽樣處理mutation
                new_population.append(Agent(new_strategy))
            self.population = new_population

            # 記錄比例
            counts = [0] * self.num_strategies
            for agent in self.population:
                counts[agent.strategy] += 1
            self.history_ratio.append([c / self.population_now for c in counts])
            self.history_num.append([c for c in counts])
            
    

    def plot_fitness_distribution(self, fitnesses, strategies, gen, num_bins=20):
        # 分箱（quantization）
        bins = np.linspace(fitnesses.min(), fitnesses.max() + 1e-6, num_bins + 1)
        bin_indices = np.digitize(fitnesses, bins) - 1  # 0-based

        # 計數矩陣 [bin, strategy] → 每個策略在每個區間的人數
        count_matrix = np.zeros((num_bins, self.num_strategies), dtype=int)
        for b, s in zip(bin_indices, strategies):
            if 0 <= b < num_bins:
                count_matrix[b, s] += 1

        # 畫圖：堆疊長條圖
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bottom = np.zeros(num_bins)
        colors = plt.cm.tab10(np.arange(self.num_strategies))  # 最多支援10種策略顏色

        plt.figure(figsize=(10, 5))
        for s in range(self.num_strategies):
            plt.bar(bin_centers, count_matrix[:, s], bottom=bottom, color=colors[s],
                    label=f"Strategy {s}", width=bin_centers[1] - bin_centers[0])
            bottom += count_matrix[:, s]

        plt.xlabel("Fitness (Shifted & Quantized)")
        plt.ylabel("Count")
        plt.title(f"Generation {gen} - Fitness Distribution by Strategy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)



    def plot_evolution(self, strategy_names=None):
        ratio_arr = np.array(self.history_ratio)
        num_arr = np.array(self.history_num)
        generations = np.arange(len(ratio_arr))

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(self.num_strategies)]

        # --- 第一張圖：策略比例 ---
        plt.figure(figsize=(10, 5))
        for i in range(self.num_strategies):
            plt.plot(generations, ratio_arr[:, i], label=strategy_names[i], linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Proportion")
        plt.title("Strategy Proportions Over Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

        # --- 第二張圖：策略絕對數量 ---
        plt.figure(figsize=(10, 5))
        for i in range(self.num_strategies):
            plt.plot(generations, num_arr[:, i], label=strategy_names[i], linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Number of Agents")
        plt.title("Strategy Counts Over Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()




# ====== Main ======
# ====== Parameters ======
R = 50 # reward
I = -100 # injury
C = -10 # cost

population_init = 5000
population_cap = 10000
logistic_r = 0.1

pairing_num = 100 # 用於固定互動次數
attack_rate = 0.0001
handling_time = 0.01
fix_pairing_num = False

generations = 40
payoff_matrix = np.array([  # payoff_matrix[i][j] = [i 得分, j 得分]
    [[R, I], [R, 0], [R, I], [R, 0]],  # Hawk random: 0, 2
    [[0, R], [R+C, C], [R+C, C], [0, R]],  # Dove random: 1, 2
    [[R, I], [R+C, C], [R+C, C], [R, 0]],  # 復仇者 random: 0, 1, 2  
    [[0, R], [R, 0], [0, R], [0, R]]  # 欺負弱小者 random: 3
])
interaction_random_matrix = np.array([
    [1, 0, 1, 0], 
    [0, 1, 1, 0], 
    [1, 1, 1, 0],
    [0, 0, 0, 1]
])
mutation_matrix = np.array([  # mutation_matrix[i][j] = i 策略突變成 j 的機率
    [0.95, 0.01, 0.02, 0.02],
    [0.01, 0.95, 0.02, 0.02],
    [0.02, 0.02, 0.94, 0.02],
    [0.02, 0.02, 0.02, 0.94]
])
initial_distribution = np.array([0.0, 1.0, 0.0, 0.0])  # 初始族群

# ====== Simulation ======
sim = EvolutionSimulator(
    payoff_matrix=payoff_matrix,
    mutation_matrix=mutation_matrix,
    interaction_random_matrix=interaction_random_matrix,
    population_init=population_init,
    population_cap=population_cap,
    logistic_r=logistic_r,
    attack_rate=attack_rate,
    handling_time=handling_time,
    pairing_num=pairing_num,
    generations=generations,
    fix_pairing_num=fix_pairing_num
)
sim.initialize_population(initial_distribution)
sim.history_ratio.append(initial_distribution) 
sim.history_num.append(initial_distribution * population_init)
sim.run()
sim.plot_evolution()

# ✅ 複雜策略（像 Tit-for-Tat、報復者） 「報償者（retaliator）」、「騙子（cheater）」
# ✅ 分組或空間分佈互動（只與鄰居互動）
# ✅ 策略學習或模仿機制
# https://chatgpt.com/c/685904cd-9d84-800e-9c28-fc3087d25124