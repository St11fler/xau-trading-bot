import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

############################################################
# 1) Зареждане на данни
############################################################

CSV_FILE = 'XAUUSD_data_multiple_timeframes.csv'
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Файлът {CSV_FILE} не е намерен. Моля, го поставете в текущата папка.")

# Зареждаме CSV файла и почистваме данните
df = pd.read_csv(CSV_FILE, index_col='time', parse_dates=True)
df.dropna(inplace=True)

# Избираме колони за наблюдение – тук вземаме пример: close_M1, pred_close_M1, rsi (или atr), според наличните данни.
columns_for_observation = [c for c in df.columns if any(sub in c for sub in ["close_M1", "pred_close_M1", "RSI", "rsi", "ATR", "atr"])]
if len(columns_for_observation) == 0:
    raise ValueError("Няма намерени колони за observation. Променете филтъра.")
data_array = df[columns_for_observation].values  # shape (N, num_features)

############################################################
# 2) Дефиниране на TradingEnv (с Gymnasium)
############################################################

class TradingEnv(gym.Env):
    """
    TradingEnv – подобрена търговска среда с реалистична симулация на търговия.
    
    Actions:
      0: Hold
      1: Buy
      2: Sell

    Observation:
      Вектор съдържащ данни от избраните колони (реални и/или прогнозирани цени + индикатори).

    Reward:
      - При отваряне/затваряне на позиция се изчислява печалба/загуба,
        като се отчитат фиксирани spread и commission.
      - При края на епизода се добавя финална корекция (баланс - initial_balance)*0.01.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0,
                 spread: float = 0.1, commission: float = 0.2):
        """
        Parameters
        ----------
        data : np.ndarray
            Масив (N, num_features) с данни за търговия.
        initial_balance : float
            Начален баланс (по подразбиране 10000).
        spread : float
            Фиксиран spread.
        commission : float
            Фиксирана комисионна за сделка.
        """
        super(TradingEnv, self).__init__()
        self.data = data
        self.n_steps = len(data) - 1
        self.current_step = 0

        self.initial_balance = initial_balance
        self.balance = initial_balance

        self.spread = spread
        self.commission = commission

        self.position = None    # 'long', 'short', или None
        self.entry_price = 0.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def _get_current_price(self) -> float:
        """Връща “реалната” цена от първата колона (например close_M1)."""
        return float(self.data[self.current_step][0])

    def _get_observation(self) -> np.ndarray:
        """Връща наблюдението за текущата стъпка като float32 вектор."""
        return self.data[self.current_step].astype(np.float32)

    def _calculate_reward(self, action: int) -> float:
        """
        Изчислява reward въз основа на зададеното действие и реализираната печалба/загуба.
        Взима предвид spread и commission.
        """
        reward = 0.0
        current_price = self._get_current_price()

        if action == 1:  # Buy
            if self.position == 'short':
                profit = (self.entry_price - current_price) - self.spread - self.commission
                self.balance += profit
                reward += profit
                self.position = None
            if self.position is None:
                self.balance -= self.commission
                reward -= self.commission
                self.position = 'long'
                self.entry_price = current_price

        elif action == 2:  # Sell
            if self.position == 'long':
                profit = (current_price - self.entry_price) - self.spread - self.commission
                self.balance += profit
                reward += profit
                self.position = None
            if self.position is None:
                self.balance -= self.commission
                reward -= self.commission
                self.position = 'short'
                self.entry_price = current_price

        # При Hold (action == 0) reward оставаме 0 (може да добавите плаващ reward при желание)

        return reward

    def reset(self, *, seed=None, options=None):
        """Нулира средата за нов епизод."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        """
        Прилага даденото действие, актуализира текущата стъпка и връща:
          (obs, reward, done, truncated, info).
        """
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = False

        if self.current_step >= self.n_steps:
            # Ако има отворена позиция, затваряме я насила
            if self.position == 'long':
                final_profit = (self._get_current_price() - self.entry_price) - self.spread - self.commission
                self.balance += final_profit
                reward += final_profit
                self.position = None
            elif self.position == 'short':
                final_profit = (self.entry_price - self._get_current_price()) - self.spread - self.commission
                self.balance += final_profit
                reward += final_profit
                self.position = None

            final_gain = self.balance - self.initial_balance
            reward += final_gain * 0.01  # Финална корекция
            done = True

        obs = self._get_observation() if not done else np.zeros_like(self.data[0], dtype=np.float32)
        info = {"balance": self.balance, "step": self.current_step}
        return obs, reward, done, False, info

    def render(self):
        """(Опционално) Визуализира състоянието."""
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position}")

############################################################
# 3) Непрекъснат тренировъчен loop с записване на логове
############################################################

def continuous_training_loop():
    """
    Изпълнява непрекъснат loop, в който:
      - Агентът се дообучава за фиксирана стъпка (напр. 10,000 timesteps).
      - След всеки епизод моделът се тества върху същата среда.
      - Записва се финалният баланс и общ reward.
      - Ако финалният баланс е по-добър от предишната най-добра стойност, се отпечатва като подобрение.
      - Всичките данни (епизод, финален баланс, общ reward) се записват в 'training_log.csv'.
      - Loop-ът не спира, докато не бъде прекратен ръчно (Ctrl+C).
    """
    # Създаваме средата
    env = TradingEnv(data_array, initial_balance=10000.0, spread=0.1, commission=0.2)
    vec_env = DummyVecEnv([lambda: env])

    # Зареждаме съществуващ модел, ако има, или създаваме нов
    model_file = "ppo_perfect_trading_agent.zip"
    if os.path.exists(model_file):
        print("Вече има обучен модел. Зареждам го...")
        model = PPO.load("ppo_perfect_trading_agent", env=vec_env)
        print("Моделът е зареден!")
    else:
        print("Създавам нов модел...")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=1e-4, n_steps=2048, batch_size=64, gamma=0.99)

    # Подготовка на лог файл
    log_filename = "training_log.csv"
    if os.path.exists(log_filename):
        log_df = pd.read_csv(log_filename)
        best_balance = log_df["final_balance"].max()
    else:
        log_df = pd.DataFrame(columns=["episode", "final_balance", "total_reward"])
        best_balance = -np.inf

    episode_counter = log_df["episode"].max() if not log_df.empty else 0

    try:
        while True:
            episode_counter += 1
            training_timesteps = 10_000
            print(f"\n--- Епизод {episode_counter}: Обучение за {training_timesteps} timesteps ---")
            model.learn(total_timesteps=training_timesteps, reset_num_timesteps=False)
            model.save("ppo_perfect_trading_agent")
            print("Моделът е записан (ppo_perfect_trading_agent.zip).")

            # Тестваме модела
            test_env = TradingEnv(data_array, initial_balance=10000.0, spread=0.1, commission=0.2)
            obs, _ = test_env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = test_env.step(action)
                total_reward += reward

            final_balance = info["balance"]
            print(f"Епизод {episode_counter}: Финален баланс = {final_balance:.2f} | Общо reward = {total_reward:.2f}")

            # Записваме логовете
            new_log = {"episode": episode_counter, "final_balance": final_balance, "total_reward": total_reward}
            log_df = log_df._append(new_log, ignore_index=True)
            log_df.to_csv(log_filename, index=False)

            # Отпечатваме подобрението, ако има такова
            if final_balance > best_balance:
                best_balance = final_balance
                print(f">>> Подобрение: Нов най-добър баланс = {final_balance:.2f}")
            else:
                print(f"Няма подобрение. Текущ най-добър баланс = {best_balance:.2f}")

            # Визуализираме историята (на всяка итерация можем да обновяваме графиката)
            plt.figure(figsize=(10, 5))
            plt.plot(log_df["episode"], log_df["final_balance"], marker='o', linestyle='-')
            plt.title("История на финалния баланс")
            plt.xlabel("Епизод")
            plt.ylabel("Финален баланс")
            plt.grid(True)
            plt.pause(0.01)  # Обновява графиката
            plt.close("all")

            # Изчакване преди следващия епизод
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nОбучението беше прекратено от потребителя. Запазвам логовете и излизам...")
        log_df.to_csv(log_filename, index=False)
        # Финална визуализация на историята
        plt.figure(figsize=(10, 5))
        plt.plot(log_df["episode"], log_df["final_balance"], marker='o', linestyle='-')
        plt.title("История на финалния баланс")
        plt.xlabel("Епизод")
        plt.ylabel("Финален баланс")
        plt.grid(True)
        plt.show()
        return

############################################################
# 4) Главна функция (main)
############################################################

def main():
    """
    Основната функция, която стартира непрекъснатия тренировъчен loop.
    """
    continuous_training_loop()

if __name__ == "__main__":
    main()