from sympy import Mul, Symbol, Pow
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_sf(rf):
    std_factor = 100 * np.std(csym_arr) - rf * np.max(csym_arr)
    sf = std_factor / (std_factor + np.average(csym_arr) * rf)
    return np.clip(sf, 0, 1)


def draw_sf_rf():
    rf = np.linspace(0, 100, 1000)
    sf_ls = []
    for rf_coeff in rf:
        sf_ls.append(get_sf(rf_coeff))

    plt.grid()
    plt.plot(sf_ls) # Y axis - sf, X axis - rf
    plt.show()


def add_array():
    additional_arr = np.random.uniform(0., 0.5, 70)
    new_array = np.concatenate((csym_arr, additional_arr))
    return new_array


# csym_arr = np.array([200, 250, 20, 30, 0.01])
# csym_arr = np.array([1, 20, 15, 30, 1])
# csym_arr = np.array([10, 20, 15, 30, 23])
csym_arr = np.array([-0.5166702263242204, 1e-06, 0.30847567753139465, -0.0021320685285910806, -0.013484108957650603, 1e-06, 0.0004800419368503562, 0.0005433419101957565, 0.0033599844393899943, 1e-06, 1e-06, 1e-06, -0.00027104834002254167, -0.001684295171560161, 1.1097813793237595e-05, 1e-06, 1e-06, 1e-06, -4.3438843718743564e-07, 2.1009290401237286e-06, -3.2298730017636938e-06, 1e-06, 1e-06, 1e-06, 1.6266320830728536e-06])
csym_arr = np.fabs(csym_arr)
# csym_arr = np.array([0.3576857683828733, 0.1412769188927505, 0.2812483095159021, 0.1414261269599306, 0.14836287624854355])

# csym_arr = np.fabs(np.array([-1, -2, 5, 0.3, 0.1]))

# relative_factor = 23.77903505027352
# relative_factor = 30
# csym_arr = add_array()
# draw_sf_rf()

# std_factor = 100 * np.std(csym_arr) - relative_factor * np.max(csym_arr)
# sf = std_factor / (std_factor + np.average(csym_arr) * relative_factor)
# np.clip(sf, 0, 1)

# smoothing_factor = 0.0
mmf = 2.4
min_max_coeff = np.max(csym_arr) - mmf*np.min(csym_arr)
smoothing_factor = min_max_coeff / (min_max_coeff + (mmf-1) * np.average(csym_arr))


print(min_max_coeff, (min_max_coeff + (mmf-1) * np.average(csym_arr)))


uniform_csym = np.array([np.sum(csym_arr) / len(csym_arr)] * len(csym_arr))

smoothed_array = (1-smoothing_factor) * csym_arr + smoothing_factor * uniform_csym
final_probabilities = smoothed_array / np.sum(smoothed_array)


fig, ax = plt.subplots(figsize=(16, 8))
ax.set_ylim(0, np.max(final_probabilities) + 0.01)#0.505)
sns.barplot(x=np.arange(len(csym_arr)), y=final_probabilities, orient="v", ax=ax)
plt.grid()
# plt.title(np.std(smoothed_array) / np.max(smoothed_array) * 100)
plt.title(f"Smoothing factor: {smoothing_factor:.3f}")
plt.show()
