
import os
import re
import matplotlib.pyplot as plt
LINE_RE = r'\d+ .  The MI train for \((\d+), (\d+), (\d+)\) box is: (\d+.\d+) \n'

with open(os.path.join('./figures', 'losses', 'entropy_calculation', 'sandnet3d', 'solid_200', 'message_entropycalc.log'), 'r') as f:
    lines = f.readlines()
    lines = [i for cntr, i in enumerate(lines) if cntr >= 6 and cntr % 2 == 0]
    results = {}
    for line in lines:
        try:
            x, y, z, mi = re.findall(LINE_RE, line)[0]
        except Exception as e:
            pass
        results[(int(x), int(y), int(z))] = float(mi)
    mi_normed = []
    Vs = []
    axis_size = 4
    for shape, mi in results.items():
        if shape[0] != axis_size or shape[1] != axis_size:
            continue
        A = shape[0] * shape[1] 
        V = shape[0] * shape[1] * shape[2]
        mi_normed.append(mi/ A)
        Vs.append(V)
    plt.plot(Vs, mi_normed, 'o')
    plt.xlabel('Volume')
    plt.ylabel('mi / Area')
    plt.title(f'x size = {axis_size}, y size = {axis_size}')
    plt.savefig('./some_figure')