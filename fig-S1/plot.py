# -*- coding: utf-8 -*-

###################################################
# path 
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

# packages
from src.utils import *

###################################################

data_path_1 = f"{folder}/data-a"
data_path_2 = f"{folder}/data-b"
data_path_3 = f"{folder}/data-c"

alphas = np.load(os.path.join(data_path_1, "alphas.npy"))
alpha_separate_mean = np.load(os.path.join(data_path_1, "separate_test_mean.npy"))
alpha_separate_std = np.load(os.path.join(data_path_1, "separate_test_std.npy"))
alpha_merged_mean = np.load(os.path.join(data_path_1, "merged_test_mean.npy"))
alpha_merged_std = np.load(os.path.join(data_path_1, "merged_test_std.npy"))

separate_train_losses_1 = np.load(os.path.join(data_path_2, "separate_train_losses.npy"))
separate_test_losses_1 = np.load(os.path.join(data_path_2, "separate_test_losses.npy"))
merged_train_losses_1 = np.load(os.path.join(data_path_2, "merged_train_losses.npy"))
merged_test_losses_1 = np.load(os.path.join(data_path_2, "merged_test_losses.npy"))

epochs = separate_train_losses_1.shape[0]

separate_train_losses_2 = np.load(os.path.join(data_path_3, "separate_train_losses.npy"))
separate_test_losses_2 = np.load(os.path.join(data_path_3, "separate_test_losses.npy"))
merged_train_losses_2 = np.load(os.path.join(data_path_3, "merged_train_losses.npy"))
merged_test_losses_2 = np.load(os.path.join(data_path_3, "merged_test_losses.npy"))

W_q_1 = np.load(os.path.join(data_path_2, "separate_W_q.npy"))
W_k_1 = np.load(os.path.join(data_path_2, "separate_W_k.npy"))
W_qk_1 = W_q_1 @ W_k_1.T

W_1 = np.load(os.path.join(data_path_2, "merged_W.npy"))

W_q_2 = np.load(os.path.join(data_path_3, "separate_W_q.npy"))
W_k_2 = np.load(os.path.join(data_path_3, "separate_W_k.npy"))
W_qk_2 = W_q_2 @ W_k_2.T

W_2 = np.load(os.path.join(data_path_3, "merged_W.npy"))

###################################################
# plot
###################################################

fig = plt.figure(figsize=cm2inch(17.9, 7.6), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2, ncols=3, 
    height_ratios=[3, 2], 
    left=0.05, right=0.75,
    hspace=0, wspace=0
    )

###################################################

ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(range(1, epochs+1), separate_train_losses_1, '--', color=colors[0], linewidth=1.2)
ax1.plot(range(1, epochs+1), separate_test_losses_1, color=colors[0], linewidth=1.2)
ax1.plot(range(1, epochs+1), merged_train_losses_1, '--', color=colors[1], linewidth=1.2)
ax1.plot(range(1, epochs+1), merged_test_losses_1, color=colors[1], linewidth=1.2)

ax1.set_ylim(-0.2, 4.2)
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epochs')

###################################################

ax2 = fig.add_subplot(gs[0, 1])

ax2.plot(alphas, alpha_separate_mean, linewidth=1.2, color=colors[0])
ax2.plot(alphas, alpha_merged_mean, linewidth=1.2, color=colors[1])

delete_index = [11, 12, 13]

alphas = np.delete(alphas, delete_index)
alpha_separate_mean = np.delete(alpha_separate_mean, delete_index)
alpha_separate_std = np.delete(alpha_separate_std, delete_index)
alpha_merged_mean = np.delete(alpha_merged_mean, delete_index)
alpha_merged_std = np.delete(alpha_merged_std, delete_index)

ax2.errorbar(alphas, alpha_separate_mean, yerr=alpha_separate_std, fmt='o', markersize=3, color=colors[0])
ax2.errorbar(alphas, alpha_merged_mean, yerr=alpha_merged_std, fmt='o', markersize=3, color=colors[1])

ax2.set_xticks([0, 1, 2])
ax2.set_xlabel(r'$\alpha=P/D^2$')
ax2.set_ylim(-0.2, 4.2)
ax2.set_yticklabels([])

###################################################

ax3 = fig.add_subplot(gs[0, 2])

ax3.plot(range(1, epochs+1), separate_train_losses_2, '--', color=colors[0], linewidth=1.2, label="training, separate")
ax3.plot(range(1, epochs+1), merged_train_losses_2, '--', color=colors[1], linewidth=1.2,
            label="training, merged")
ax3.plot(range(1, epochs+1), separate_test_losses_2, color=colors[0], linewidth=1.2, label="test, separate")
ax3.plot(range(1, epochs+1), merged_test_losses_2, color=colors[1], linewidth=1.2, label="test, merged")

ax3.set_ylim(-0.2, 4.2)
ax3.set_yticklabels([])
ax3.set_xlabel('Epochs')
ax3.legend(fontsize=8)

###################################################

ax1.text(-0.08, 1, '(a)', transform=ax1.transAxes, fontsize=10, va='top', ha='right')
ax2.text(-0.03, 1, '(b)', transform=ax2.transAxes, fontsize=10, va='top', ha='right')
ax3.text(-0.03, 1, '(c)', transform=ax3.transAxes, fontsize=10, va='top', ha='right')

###################################################

gs = fig.add_gridspec(
    nrows=2, ncols=13, 
    width_ratios=[2, 5, 5, 5, 0.5, 5, 2, 5, 5, 5, 0.5, 5, 3],
    height_ratios=[2, 1], 
    left=0.6, right=0.95, 
    hspace=0, wspace=0
    )

###################################################

ax4 = fig.add_subplot(gs[1, 1])

ax4.imshow(W_q_1, cmap='coolwarm', vmin=-1, vmax=1)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(0.5, -0.08, r'$\mathbf{W}_{\mathrm{Q}}$', transform=ax4.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax5 = fig.add_subplot(gs[1, 2])

ax5.imshow(W_k_1, cmap='coolwarm', vmin=-1, vmax=1)
ax5.set_xticks([])
ax5.set_yticks([])

ax5.text(0.5, -0.08, r'$\mathbf{W}_{\mathrm{K}}$', transform=ax5.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax6 = fig.add_subplot(gs[1, 3])

ax6.imshow(W_qk_1, cmap='coolwarm', vmin=-1, vmax=1)
ax6.set_xticks([])
ax6.set_yticks([])

ax6.text(0.5, -0.05, r'$\mathbf{W}_{\mathrm{Q}}^\top \mathbf{W}_{\mathrm{K}}$', transform=ax6.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax7 = fig.add_subplot(gs[1, 5])

ax7.imshow(W_1, cmap='coolwarm', vmin=-1, vmax=1)
ax7.set_xticks([])
ax7.set_yticks([])

ax7.text(0.5, -0.07, r'$\mathbf{W}$', transform=ax7.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax8 = fig.add_subplot(gs[1, 7])

ax8.imshow(W_q_2, cmap='coolwarm', vmin=-1, vmax=1)
ax8.set_xticks([])
ax8.set_yticks([])

ax8.text(0.5, -0.09, r'$\mathbf{W}_{\mathrm{Q}}$', transform=ax8.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax9 = fig.add_subplot(gs[1, 8])

ax9.imshow(W_k_2, cmap='coolwarm', vmin=-1, vmax=1)
ax9.set_xticks([])
ax9.set_yticks([])

ax9.text(0.5, -0.09, r'$\mathbf{W}_{\mathrm{K}}$', transform=ax9.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax10 = fig.add_subplot(gs[1, 9])

ax10.imshow(W_qk_2, cmap='coolwarm', vmin=-1, vmax=1)
ax10.set_xticks([])
ax10.set_yticks([])

ax10.text(0.5, -0.05, r'$\mathbf{W}_{\mathrm{Q}}^\top \mathbf{W}_{\mathrm{K}}$', transform=ax10.transAxes, fontsize=10, va='top', ha='center')

###################################################

ax11 = fig.add_subplot(gs[1, 11])

ax11.imshow(W_2, cmap='coolwarm', vmin=-1, vmax=1)
ax11.set_xticks([])
ax11.set_yticks([])

ax11.text(0.5, -0.08, r'$\mathbf{W}$', transform=ax11.transAxes, fontsize=10, va='top', ha='center')

###################################################

cbar_ax = fig.add_subplot(gs[1, 12])
cbar = fig.colorbar(ax11.images[0], cax=cbar_ax)
cbar.set_ticks([-1, 0, 1])
cbar_ax.set_position([cbar_ax.get_position().x0 + 0.03, cbar_ax.get_position().y0 - 0.09, 0.012, 0.3])

#############################################

rect_1 = plt.Rectangle((0.036, 0.005), 0.438, 0.313, transform=fig.transFigure, color="black", fill=False, linewidth=0.8, linestyle='--')
fig.patches.append(rect_1)

fig.text(0.006, 0.3215, '(d)', fontsize=10, va='top', ha='left', transform=fig.transFigure)

rect_2 = plt.Rectangle((0.5075, 0.005), 0.438, 0.313, transform=fig.transFigure, color="black", fill=False, linewidth=0.8, linestyle='--')
fig.patches.append(rect_2)

fig.text(0.4805, 0.3215, '(e)', fontsize=10, va='top', ha='left', transform=fig.transFigure)

#############################################

rect_3 = plt.Rectangle((0.4145, 0.65), 0.013, 0.2, transform=fig.transFigure, color="black", fill=False, linewidth=0.6, linestyle='-')
fig.patches.append(rect_3)

line_1 = plt.Line2D([0.4155, 0.336], [0.85, 0.985], transform=fig.transFigure, color="black", linewidth=0.6, linestyle='-')
fig.lines.append(line_1)

line_2 = plt.Line2D([0.4155, 0.336], [0.65, 0.478], transform=fig.transFigure, color="black", linewidth=0.6, linestyle='-')
fig.lines.append(line_2)

#############################################

rect_4 = plt.Rectangle((0.6185, 0.5), 0.013, 0.1, transform=fig.transFigure, color="black", fill=False, linewidth=0.6, linestyle='-')
fig.patches.append(rect_4)

line_3 = plt.Line2D([0.632, 0.7075], [0.6, 0.985], transform=fig.transFigure, color="black", linewidth=0.6, linestyle='-')
fig.lines.append(line_3)

line_4 = plt.Line2D([0.632, 0.7075], [0.5, 0.478], transform=fig.transFigure, color="black", linewidth=0.6, linestyle='-')
fig.lines.append(line_4)

#############################################

plt.savefig(f"{folder}/{folder}.pdf")
