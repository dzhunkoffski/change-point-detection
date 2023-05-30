import numpy as np
from typing import *

import matplotlib.pyplot as plt
import seaborn as sns
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

def find_change_points(ts, min_stability_duration: int):
    last_stable_level = round(ts[0] / np.pi)
    last_unstable_level = 0
    last_stable_ix = 0
    last_unstable_ix = 0
    
    stability_duration = 0
    
    change_points = []
    
    for i, xt in enumerate(ts):
        if i == 0:
            continue
        if round(xt / np.pi) % 2 == 0:
            # unstable уровень - пока здесь ничего не делаю
            last_unstable_level = round(xt / np.pi)
            last_unstable_ix = i
        else:
            # stable
            if last_stable_level != round(xt / np.pi):
                # прыгнули на новый устойчивый уровень -> был change_point
                if stability_duration >= min_stability_duration:
                    # добавляем change point только если уровень устойчивости длился больше некоторого T
                    change_points.append(last_stable_ix)
                stability_duration = 1
                last_stable_level = round(xt / np.pi)
                last_stable_ix = i
            else:
                # остались на прежнем уровне, либо вернулись с неустойчивой точки
                last_stable_ix = i
                stability_duration += 1
    return change_points

def get_windows(cp_ixs, ts_len):
    cp_ixs = [0] + cp_ixs + [ts_len-1]
    win = [(cp_ixs[i] - cp_ixs[i-1]) for i in range(1, len(cp_ixs))]
    return np.array(win)

def get_ts_cps(ts_params, step_size, min_stability_duration):
    integralAlgo = IntegralAlgorithm(**ts_params)
    ts, _ = integralAlgo.generate(step_size)
    cp_ixs = find_change_points(ts, min_stability_duration)
    return ts, cp_ixs

def demonstrate(ts_params, step_size, ts, cp_ixs, bins=75, max_winlen=None, subseq_len=1000):
    ts = ts / np.pi
    wins = get_windows(cp_ixs, len(ts))
    cp_ixs = np.array(cp_ixs)

    fig, axs = plt.subplots(6, 1, figsize=(16, 24))
    axs[0].plot(np.arange(0, step_size * len(ts), step_size), ts)
    axs[0].set_title('Time serie')
    axs[0].set_xlabel('Time', loc='right')
    axs[0].set_ylabel(r'$x_t$ in $\pi$ scale')

    if max_winlen is not None:
        wins = wins[np.where(wins <= max_winlen)]
    axs[1].hist(wins, bins=bins)
    axs[1].set_title(f'Steps between CPs distribution. Total CPs: {len(cp_ixs)}. Mean: {round(wins.mean(), 2)}. Median: {np.median(wins)}')
    for i in range(2, len(axs)):
        LB = np.random.choice(len(ts) - 2000)
        RB = LB + subseq_len
        axs[i].set_title(f'Zoomed time serie from {LB} to {RB}')
        axs[i].plot(np.arange(LB, RB), ts[LB:RB], label='Time serie')
        use_ixs = cp_ixs[np.where(LB <= cp_ixs)]
        use_ixs = use_ixs[np.where(use_ixs < RB)]
        axs[i].vlines(use_ixs, ymin=ts[LB:RB].min(), ymax=ts[LB:RB].max(), 
                      color='black', linestyles='dashed', label='Change points')
        axs[i].set_xlabel('Time', loc='right')
        axs[i].set_ylabel(r'$x_t$ in $\pi$ scale')
        axs[i].legend()
    
    return ts * np.pi

def demonstrate_result(ts, cp_ixs, probs, T, val=False, seq_len=1000):
    fig, axs = plt.subplots(6, 1, figsize=(16, 4 * 6))
    ts = ts / np.pi
    if val == True:
        cp_ixs -= T
    for i in range(6):
        LB = np.random.choice(len(ts) - seq_len - 1)
        RB = LB + seq_len
        axs[i].set_title(f'Zoomed time serie from {LB} to {RB}')
        axs[i].plot(np.arange(LB, RB), ts[LB:RB], label='Time serie')
        use_ixs = cp_ixs[np.where(LB <= cp_ixs)]
        use_ixs = use_ixs[np.where(use_ixs < RB)]
        axs[i].vlines(use_ixs, ymin=ts[LB:RB].min(), ymax=ts[LB:RB].max(), 
                      color='black', linestyles='dashed', label='Change points')
        axs[i].set_xlabel('Time', loc='right')
        axs[i].set_ylabel(r'$x_t$ in $\pi$ scale')
        axs[i].legend()

        ax2 = axs[i].twinx()
        ax2.scatter(np.arange(LB, RB), probs[LB:RB], color='red')
    ts = ts * np.pi

def show_zoomed_result(ts, cp_ixs, probs, T, LB, RB, val=False):
    ts = ts / np.pi
    fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    if val == True:
        cp_ixs -= T
    axs.set_title(f'Zoomed timeserie from {LB} to {RB}')
    axs.plot(np.arange(LB, RB), ts[LB:RB], label='Time serie')
    use_ixs = cp_ixs[np.where(LB <= cp_ixs)]
    use_ixs = use_ixs[np.where(use_ixs < RB)]
    axs.vlines(use_ixs, ymin=ts[LB:RB].min(), ymax=ts[LB:RB].max(), 
                  color='black', linestyles='dashed', label='Change points')
    axs.set_xlabel('Time', loc='right')
    axs.set_ylabel(r'$x_t$ in $\pi$ scale')
    axs.legend()

    ax2 = axs.twinx()
    ax2.plot(np.arange(LB, RB), probs[LB:RB], color='red')
    ts = ts * np.pi

def white_noise_generator(N: int, step_size: int, D: float):
    noise = []
    for _ in range(N):
        a = np.random.rand()
        b = np.random.rand()
        noise.append(
            np.sqrt(-4 * D * step_size * np.log(a)) * np.cos(2 * np.pi * b)
        )
    return np.array(noise)

def colored_noise_generator(N: int, step_size: int, D: float, lambd: float):
    m = np.random.rand()
    n = np.random.rand()
    noise = []
    noise.append(np.sqrt(-2*D*lambd*np.log(m))*np.cos(2*np.pi*n))
    for _ in range(1, N):
        a = np.random.rand()
        b = np.random.rand()
        r = np.sqrt(-2*D*lambd*(1 - np.exp(-lambd*step_size))*np.log(a)) * np.cos(2*np.pi*b)
        noise.append(
            noise[-1] * np.exp(-lambd * step_size) + r
        )
    return np.array(noise)

class IntegralAlgorithm:
    def __init__(self, x0: float, lambd: float, n_steps: int, alpha: float):
        self.x0 = x0
        self.lambd = lambd
        self.n_steps = n_steps
        self.alpha = alpha
        self.a = [np.random.rand() for _ in range(self.n_steps - 1)]
        self.b = [np.random.rand() for _ in range(self.n_steps - 1)]

        self.m = np.random.rand()
        self.n = np.random.rand()

    def step(self, step_size: float):
        x = self.timeserie[-1]
        eps = self.noise[-1]

        p = np.sin(x) + self.alpha * eps
        self.timeserie.append(x + step_size * p)
        E = np.exp(-self.lambd * step_size)
        h = np.sqrt(-2 * (self.alpha ** 2) * self.lambd * (1 - (E ** 2)) * np.log(self.a[self.step_no])) * np.cos(2 * np.pi * self.b[self.step_no])
        self.noise.append(eps * E + h)

        self.step_no += 1
    
    def generate(self, step_size: float):
        self.step_no = 0
        self.noise = [np.sqrt(-2 * (self.alpha ** 2) * self.lambd * np.log(self.m)) * np.cos(2 * np.pi * self.n)]
        self.timeserie = [self.x0]
        for _ in range(1, self.n_steps):
            self.step(step_size)
        return np.array(self.timeserie), np.array(self.noise)
