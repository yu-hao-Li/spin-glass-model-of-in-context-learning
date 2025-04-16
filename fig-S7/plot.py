# -*- coding: utf-8 -*-

###################################################
# path
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

data_path = f"{folder}/data"

# packages
from src.utils import *

###################################################

P_list = [100, 500, 1500]
zero_ratios = np.load(os.path.join(data_path, "zero-ratio.npy")).T

fig, ax = plt.subplots(figsize=cm2inch(8, 6), constrained_layout=True)

colors_1 = ["#A3C9D5", "#ECB477", "#C2ABC8"]
colors_2 = ["#F2D086", "#E2CD89", "#C3BC6C", "#B1C07A", "#95BE7E", "#7CBC8A", "#43BC97"]

for i, P in enumerate(P_list):

   eigenvalues_list = np.load(os.path.join(data_path, f"P={P}.npy"))
   
   ax.hist(eigenvalues_list, bins=500, density=True, alpha=0.9, label=f"P={P}", log=True, color=colors_1[i])

   ax.set_xlim(-0.1, 5.1)
   ax.set_xlabel("Eigenvalues", labelpad=2)
   ax.set_ylabel("Density", labelpad=2)
   ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1), fontsize=7, frameon=False)

   ax3d = fig.add_axes([0.45, 0.43, 0.6, 0.6], projection='3d')
   ax3d.set_facecolor('none')

   D_list_3d = np.linspace(10, 40, 7).astype(int)
   P_list_3d = np.linspace(25, 2000, 80).astype(int)

   for i, D in enumerate(D_list_3d):
      ax3d.plot([D] * len(P_list_3d), P_list_3d, zero_ratios[:, i], label=f'D={D}', color=colors_2[i], linewidth=0.8)

   for i, P in enumerate(P_list):
      zero_ratio = zero_ratios[P_list_3d.tolist().index(P), D_list_3d.tolist().index(20)]
      ax3d.scatter(20, P, zero_ratio, color=colors_1[i], marker='*', s=20, label=f'P={P}')
   
   ax3d.xaxis._axinfo['grid'].update({'color': 'lightgray'})
   ax3d.yaxis._axinfo['grid'].update({'color': 'lightgray'})
   ax3d.zaxis._axinfo['grid'].update({'color': 'lightgray'})

   ax3d.set_xlabel(r'$D$', labelpad=-8, fontsize=8)
   ax3d.invert_xaxis() 
   ax3d.set_xlim(43, 8)
   ax3d.set_xticks([10, 20, 30, 40])
   ax3d.set_xticklabels([10, 20, 30, 40], fontsize=8)
   ax3d.tick_params(axis='x', pad=-5)

   ax3d.set_ylabel(r'$P$', labelpad=-7, fontsize=8)
   ax3d.set_ylim(0, 2200)
   ax3d.set_yticks([1000, 2000])
   ax3d.set_yticklabels([1000, 2000], fontsize=8)
   ax3d.tick_params(axis='y', pad=-5)
   
   ax3d.set_zlim(0, 1)
   ax3d.set_zticks([0, 0.5, 1])
   ax3d.set_zticklabels([0, 0.5, 1], fontsize=8)
   ax3d.tick_params(axis='z', pad=-4)
   ax3d.view_init(elev=15, azim=35)

   ax.text(0.465, 0.72, 'Zero Ratio', transform=fig.transFigure, fontsize=8, rotation=90, ha='center', va='center')

plt.savefig(f"{folder}/{folder}.pdf")
