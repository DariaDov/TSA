import os
import numpy as np
import matplotlib.pyplot as plt

def white_noise(n):
    rng = np.random.default_rng(seed=1)
    return rng.standard_normal(n)

def simulate_arma(coefs, p, q, n):
    r_n = 50
    a0, a1, a2, a3, b1, b2, b3 = coefs
    a = [a1, a2, a3][:p]
    b = [b1, b2, b3][:q]
    total = n + r_n

    v = white_noise(total)
    y = np.zeros(total)

    for k in range(total):
        arma = a0 + v[k]
        for i, ai in enumerate(a, start=1):
            if k - i >= 0:
                arma += ai * y[k - i]
        for j, bj in enumerate(b, start=1):
            if k - j >= 0:
                arma += bj * v[k - j]
        y[k] = arma
        
    return y[r_n:], v[r_n:]

def build_regressors(y, v, p, q):
    m = max(p, q)
    rows = []
    for k in range(m, len(y)):
        row = [1.0]
        row += [y[k - i] for i in range(1, p + 1)]
        row += [v[k]]
        row += [v[k - j] for j in range(1, q + 1)]
        rows.append(row)

    return np.array(rows), y[m:]

def mnk(x, y):
    xt = x.T
    theta = np.linalg.inv(xt @ x) @ xt @ y
    return theta, y - x @ theta

def rmnk(X, Y):
    beta = 500.0

    r, c = X.shape
    P = beta * np.eye(c)
    theta = np.zeros((c, 1))
 
    hist = np.zeros((r, c))
    e = np.zeros(r)
 
    for i in range(r):
        x_row = X[i].reshape(1, -1)
        x_col = x_row.T                     
        d = 1.0 + (x_row @ P @ x_col)[0, 0]
        P = P - (P @ x_col @ x_row @ P) / (d)
        e[i] = Y[i] - (x_row @ theta)[0, 0]
        theta = theta + P @ x_col * e[i]
        hist[i] = theta.ravel()
 
    return theta.ravel(), e, hist

def metrics(Y, e, p, q):
    n = len(Y)
    n_par = p + q + 1
    
    S = float(np.sum(e ** 2))
    R2 = float(np.var(Y - e) / np.var(Y))
    IKA = n * np.log(S) + (2 * n_par)
    return S, R2, IKA

def load_data_from_files(v_filename, y_filename):
    if not os.path.exists(v_filename):
        raise FileNotFoundError(f"Файл '{v_filename}' не знайдено.")
    if not os.path.exists(y_filename):
        raise FileNotFoundError(f"Файл '{y_filename}' не знайдено.")

    v = np.loadtxt(v_filename)
    y = np.loadtxt(y_filename)

    if len(v) != len(y):
        print(f"Файл v містить {len(v)} значень, а файл y — {len(y)} значень. Довжини різні.")
        min_len = min(len(y), len(v))
        y = y[:min_len]
        v = v[:min_len]

    return v, y

def param_names(p, q):
    return (["a0"] + [f"a{i}" for i in range(1, p + 1)] + ["coef"] + [f"b{j}" for j in range(1, q + 1)])
 
def true_params(coefs, p, q):
    a0, a1, a2, a3, b1, b2, b3 = coefs
    return np.array([a0] + [a1, a2, a3][:p] + [1.0] + [b1, b2, b3][:q])

coefs = [0.0, -0.045, -0.079, 0.525, 0.3, 0.4, 0.3]
p = 3
q = 3
N = 300

mode = input("Завантаживти файли v.txt та y.txt? (y/n): ")

if mode == "n":
    print("a0,a1,a2,a3,b1,b2,b3 =", coefs)
    y, v = simulate_arma(coefs, p, q, N)

elif mode == "y":
    v_file = "/home/daria/АЧР/Labs/Test+/v.txt"
    y_file = "/home/daria/АЧР/Labs/Test+/y.txt"
    v, y = load_data_from_files(v_file, y_file)
    print(f"N = {len(y)}\n")

p_list = [1, 2, 3]
q_list = [1, 2, 3]

results = {}

for p in p_list:
    for q in q_list:
        X, Y = build_regressors(y, v, p, q)
        th_mnk, e_mnk = mnk(X, Y)
        th_rmnk, e_rmnk, hist = rmnk(X, Y)

        results[(p, q)] = dict(mnk=(th_mnk, e_mnk, metrics(Y, e_mnk, p, q)), rmnk=(th_rmnk, e_rmnk, metrics(Y, e_rmnk, p, q)), hist=hist)

print(f"{'Модель':<10}{'Метод':<7}{'S':>12}{'R2':>9}{'IKA':>12}")
print("-" * 50)
for (p, q), r in results.items():
    for name, title in (("mnk", "МНК"), ("rmnk", "РМНК")):
        S, R2, IKA = r[name][2]
        print(f"АРКС({p},{q}) {title:<7}{S:12.3f}{R2:9.4f}{IKA:12.2f}")
print()

for name, title in (("mnk", "МНК"), ("rmnk", "РМНК")):
    key = min(results, key=lambda k: results[k][name][2][2])
    print(f"Найкраща модель за IKA ({title}): АРКС{key}, "f"IKA = {results[key][name][2][2]:.2f}")
print()

r = results[(p, q)]
names = param_names(p, q)
tp = true_params(coefs, p, q) if coefs is not None else None
hist = r["hist"]
plot_idx = [i for i, n in enumerate(names) if n != "coef"]
n_plot = len(plot_idx)
cols = 3
rows_ = int(np.ceil(n_plot / cols))
fig1, axs = plt.subplots(rows_, cols, figsize=(12, 3 * rows_), squeeze=False)
for pos in range(rows_ * cols):
    ax = axs[pos // cols][pos % cols]
    if pos >= n_plot:
        ax.axis("off")
        continue
    i = plot_idx[pos]
    ax.plot(hist[:, i], label="РМНК")
    ax.axhline(r["mnk"][0][i], color="g", ls=":", label="МНК")
    if tp is not None:
        ax.axhline(tp[i], color="r", ls="--", label="Реальне значення")
    ax.set_title(names[i])
    ax.set_xlabel("k")
    ax.grid(True)
    if pos == 0:
        ax.legend()
fig1.suptitle(f"Перехідний процес оцінок")
fig1.tight_layout()

keys = list(results.keys())
x = np.arange(len(keys))
labels = [f"({p},{q})" for p, q in keys]
fig2, axs2 = plt.subplots(1, 3, figsize=(14, 4))
for ax, idx, title in zip(axs2, (1, 0, 2), ("R²", "S", "IKA")):
    ax.plot(x, [results[k]["mnk"][2][idx] for k in keys], "o-", label="МНК")
    ax.plot(x, [results[k]["rmnk"][2][idx] for k in keys], "o-", label="РМНК")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
fig2.tight_layout()
plt.show()