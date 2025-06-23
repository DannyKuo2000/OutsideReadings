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
                 population_size=200, pairing_num=100, generations=100):
        self.payoff_matrix = payoff_matrix  # shape (n, n, 2)
        self.mutation_matrix = mutation_matrix  # shape (n, n)
        self.interaction_random_matrix = interaction_random_matrix
        self.population_size = population_size
        self.pairing_num = pairing_num
        self.generations = generations
        self.num_strategies = payoff_matrix.shape[0]
        self.population = []
        self.history = []

    def initialize_population(self, initial_distribution):
        self.population = []
        counts = (initial_distribution * self.population_size).astype(int)

        # 為了避免總和不精確造成個體數不足，多出的人補給比例最高的策略
        remainder = self.population_size - np.sum(counts)
        if remainder > 0:
            max_idx = np.argmax(initial_distribution)
            counts[max_idx] += remainder

        # 根據比例創建個體
        for strategy_index, count in enumerate(counts):
            self.population.extend([Agent(strategy_index) for _ in range(count)])
        random.shuffle(self.population)

    def run(self):
        for gen in trange(self.generations, desc="Simulating Generations"):
            for agent in self.population:
                agent.fitness = 0

            # 進行互動(隨機配對)
            for _ in range(self.population_size * self.pairing_num):
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
                fitnesses = fitnesses - min_fit  # shift to make min = 0

            # === 分布視覺化（每50代畫一次） ===
            if gen % 5 == 0 or gen == self.generations - 1:
                # 分箱（quantization）
                num_bins = 20
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
                colors = plt.cm.tab10(np.arange(self.num_strategies))  # 最多10種顏色

                plt.figure(figsize=(10, 5))
                for s in range(self.num_strategies):
                    plt.bar(bin_centers, count_matrix[:, s], bottom=bottom, color=colors[s], label=f"Strategy {s}", width=bin_centers[1] - bin_centers[0])
                    bottom += count_matrix[:, s]

                plt.xlabel("Fitness (Shifted & Quantized)")
                plt.ylabel("Count")
                plt.title(f"Generation {gen} - Fitness Distribution by Strategy")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(1)

            total_fitness = fitnesses.sum()
            if total_fitness == 0:
                # 所有 fitness 一樣（或都是負的），隨機選擇
                probs = None
            else:
                probs = fitnesses / total_fitness
            # 加權隨機抽樣: k: 抽出的個數
            selected = random.choices(self.population, weights=probs, k=self.population_size) 

            new_population = []
            for agent in selected:
                i = agent.strategy
                new_strategy = np.random.choice(self.num_strategies, p=self.mutation_matrix[i])
                new_population.append(Agent(new_strategy))
            self.population = new_population

            # 記錄比例
            counts = [0] * self.num_strategies
            for agent in self.population:
                counts[agent.strategy] += 1
            self.history.append([c / self.population_size for c in counts])

    def plot(self, strategy_names=None):
        arr = np.array(self.history)
        generations = np.arange(len(arr))

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(self.num_strategies)]

        plt.figure(figsize=(10, 6))
        for i in range(self.num_strategies):
            plt.plot(generations, arr[:, i], label=strategy_names[i], linewidth=2)

        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Proportion", fontsize=12)
        plt.title("Strategy Evolution Over Time", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



# ====== Main ======
# 參數
R = 50 # reward
I = -100 # injury
C = -10 # cost

# payoff_matrix[i][j] = [i 得分, j 得分]
payoff_matrix = np.array([
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

# mutation_matrix[i][j] = i 策略突變成 j 的機率
mutation_matrix = np.array([
    [0.91, 0.03, 0.03, 0.03],
    [0.03, 0.91, 0.03, 0.03],
    [0.03, 0.03, 0.91, 0.03],
    [0.03, 0.03, 0.03, 0.91]
])

# 初始族群
initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])

# 模擬
sim = EvolutionSimulator(
    payoff_matrix=payoff_matrix,
    mutation_matrix=mutation_matrix,
    interaction_random_matrix=interaction_random_matrix,
    population_size=2000,
    pairing_num=100,
    generations=30
)

sim.initialize_population(initial_distribution)
sim.history.append(initial_distribution)
sim.run()
sim.plot()


# ✅ 突變率（如每代 1% 改策略）
# ✅ 複雜策略（像 Tit-for-Tat、報復者） 「報償者（retaliator）」、「騙子（cheater）」
# ✅ 分組或空間分佈互動（只與鄰居互動）
# ✅ 策略學習或模仿機制