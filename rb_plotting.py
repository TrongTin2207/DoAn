import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import os

def plot_rb_assignments(z_ib_sk, num_slices, num_UEs, num_RUs, num_RBs, slices, 
                       frame_num=None, time_slot=None, save_path="./result", 
                       show_plot=True, save_plot=True):
    # --- Defensive shape checks ---
    z_shape = z_ib_sk.shape
    if len(z_shape) != 4:
        raise ValueError(f"z_ib_sk must be 4D (RU, RB, slice, UE), got shape {z_shape}")
    num_RUs, num_RBs, num_slices_z, num_UEs_z = z_shape
    # Clip to min of provided and expected
    num_slices = min(num_slices, num_slices_z)
    num_UEs = min(num_UEs, num_UEs_z)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Create title based on whether it's long-term or short-term
    if time_slot is not None:
        title = f'Short-term RB Assignments - Frame {frame_num}, Time Slot {time_slot}'
    else:
        title = f'Long-term RB Assignments - Frame {frame_num}'
    
    if frame_num is None and time_slot is None:
        title = 'Resource Block Assignments'
        
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Color maps for different slices
    colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
    slice_colors = {slices[s]: colors[s % len(colors)] for s in range(num_slices)}
    
    # Plot 1: RB allocation heatmap for each slice
    ax1 = axes[0, 0]
    rb_usage_per_slice = np.zeros((num_slices, num_RBs))
    
    for s in range(num_slices):
        for b in range(num_RBs):
            if s < num_slices_z:
                rb_usage_per_slice[s, b] = np.sum(z_ib_sk[:, b, s, :num_UEs])
    
    im1 = ax1.imshow(rb_usage_per_slice, cmap='YlOrRd', aspect='auto')
    ax1.set_title('RB Usage per Slice', fontweight='bold')
    ax1.set_xlabel('Resource Block Index')
    ax1.set_ylabel('Slice Type')
    ax1.set_yticks(range(num_slices))
    ax1.set_yticklabels(slices)
    plt.colorbar(im1, ax=ax1, label='Usage Count')
    
    # Plot 2: User-RB assignment matrix for first RU
    ax2 = axes[0, 1]
    user_rb_matrix = np.zeros((num_UEs, num_RBs))
    
    for k in range(num_UEs):
        for b in range(num_RBs):
            for s in range(num_slices):
                if s < num_slices_z and k < num_UEs_z:
                    if z_ib_sk[0, b, s, k] > 0:
                        user_rb_matrix[k, b] = s + 1  # Color by slice type
    
    im2 = ax2.imshow(user_rb_matrix, cmap='tab10', aspect='auto', vmin=0, vmax=num_slices)
    ax2.set_title('User-RB Assignment (RU 0)', fontweight='bold')
    ax2.set_xlabel('Resource Block Index')
    ax2.set_ylabel('User Index')
    
    # Add grid for better visibility
    ax2.set_xticks(np.arange(-0.5, num_RBs, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, num_UEs, 1), minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Plot 3: RB utilization across RUs
    ax3 = axes[1, 0]
    ru_rb_utilization = np.zeros((num_RUs, num_RBs))
    
    for i in range(num_RUs):
        for b in range(num_RBs):
            ru_rb_utilization[i, b] = np.sum(z_ib_sk[i, b, :, :])
    
    im3 = ax3.imshow(ru_rb_utilization, cmap='viridis', aspect='auto')
    ax3.set_title('RB Utilization per RU', fontweight='bold')
    ax3.set_xlabel('Resource Block Index')
    ax3.set_ylabel('RU Index')
    plt.colorbar(im3, ax=ax3, label='Total Assignments')
    
    # Plot 4: Bar chart showing RB distribution per slice
    ax4 = axes[1, 1]
    rb_per_slice = [np.sum(rb_usage_per_slice[s, :]) for s in range(num_slices)]
    bars = ax4.bar(slices, rb_per_slice, color=['skyblue', 'lightcoral', 'lightgreen'][:num_slices])
    ax4.set_title('Total RB Assignments per Slice', fontweight='bold')
    ax4.set_ylabel('Total RB Assignments')
    ax4.set_xlabel('Slice Type')
    
    # Add value labels on bars
    for bar, value in zip(bars, rb_per_slice):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = f"rb_assignments"
        if frame_num is not None:
            filename += f"_frame{frame_num}"
        if time_slot is not None:
            filename += f"_timeslot{time_slot}"
        filename += ".png"
        
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"RB assignment plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_rb_efficiency_metrics(z_ib_sk, p_ib_sk, gain, total_R_sk, num_slices, num_UEs, 
                              num_RUs, num_RBs, slices, rb_bandwidth, frame_num=None, 
                              time_slot=None, save_path="./result", show_plot=True, save_plot=True):
 
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Create title based on whether it's long-term or short-term
    if time_slot is not None:
        title = f'Short-term RB Efficiency Metrics - Frame {frame_num}, Time Slot {time_slot}'
    else:
        title = f'Long-term RB Efficiency Metrics - Frame {frame_num}'
    
    if frame_num is None and time_slot is None:
        title = 'RB Efficiency Metrics'
        
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Spectral efficiency per RB
    ax1 = axes[0, 0]
    spectral_eff = np.zeros(num_RBs)
    
    for b in range(num_RBs):
        total_rate_rb = 0
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    if z_ib_sk[i, b, s, k] > 0:
                        snr = p_ib_sk[i, b, s, k] * gain[i, s, k, b] if gain.ndim == 4 else p_ib_sk[i, b, s, k] * gain[i, k, b]
                        rate = rb_bandwidth * np.log2(1 + snr)
                        total_rate_rb += rate
        spectral_eff[b] = total_rate_rb / rb_bandwidth
    
    ax1.bar(range(num_RBs), spectral_eff, color='steelblue', alpha=0.7)
    ax1.set_title('Spectral Efficiency per RB', fontweight='bold')
    ax1.set_xlabel('Resource Block Index')
    ax1.set_ylabel('Spectral Efficiency (bps/Hz)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power utilization per RB
    ax2 = axes[0, 1]
    power_per_rb = np.zeros(num_RBs)
    
    for b in range(num_RBs):
        power_per_rb[b] = np.sum(p_ib_sk[:, b, :, :])
    
    ax2.bar(range(num_RBs), power_per_rb, color='coral', alpha=0.7)
    ax2.set_title('Power Allocation per RB', fontweight='bold')
    ax2.set_xlabel('Resource Block Index')
    ax2.set_ylabel('Total Power (mW)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rate achievement per slice
    ax3 = axes[1, 0]
    if total_R_sk is not None:
        rates_per_slice = [np.sum(total_R_sk[s, :]) for s in range(num_slices)]
        # Chú ý total_R_sk là int, không thể thực hiện theo cách này
        bars = ax3.bar(slices, rates_per_slice, color=['lightblue', 'lightpink', 'lightgreen'][:num_slices])
        ax3.set_title('Total Rate per Slice', fontweight='bold')
        ax3.set_ylabel('Total Rate (bps)')
        ax3.set_xlabel('Slice Type')
        
        # Add value labels
        for bar, value in zip(bars, rates_per_slice):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: RB utilization efficiency
    ax4 = axes[1, 1]
    rb_utilization = []
    total_rbs = num_RUs * num_RBs
    
    for s in range(num_slices):
        used_rbs = np.sum(z_ib_sk[:, :, s, :] > 0)
        utilization = (used_rbs / total_rbs) * 100
        rb_utilization.append(utilization)
    
    bars = ax4.bar(slices, rb_utilization, color=['gold', 'orange', 'red'][:num_slices])
    ax4.set_title('RB Utilization per Slice', fontweight='bold')
    ax4.set_ylabel('Utilization (%)')
    ax4.set_xlabel('Slice Type')
    ax4.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, value in zip(bars, rb_utilization):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = f"rb_efficiency"
        if frame_num is not None:
            filename += f"_frame{frame_num}"
        if time_slot is not None:
            filename += f"_timeslot{time_slot}"
        filename += ".png"
        
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"RB efficiency plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_long_term_vs_short_term_comparison(long_term_data, short_term_data, num_slices, 
                                           slices, frame_num, save_path="./result", 
                                           show_plot=True, save_plot=True):
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Long-term vs Short-term Comparison - Frame {frame_num}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: RB utilization comparison
    ax1 = axes[0, 0]
    long_term_util = [np.sum(long_term_data['rb_usage_per_slice'][s, :]) for s in range(num_slices)]
    short_term_util = [np.sum(short_term_data['rb_usage_per_slice'][s, :]) for s in range(num_slices)]
    
    x = np.arange(len(slices))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, long_term_util, width, label='Long-term', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, short_term_util, width, label='Short-term', color='lightcoral', alpha=0.8)
    
    ax1.set_title('RB Utilization Comparison', fontweight='bold')
    ax1.set_xlabel('Slice Type')
    ax1.set_ylabel('Total RB Assignments')
    ax1.set_xticks(x)
    ax1.set_xticklabels(slices)
    ax1.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Power consumption comparison
    ax2 = axes[0, 1]
    long_term_power = long_term_data.get('total_power', 0)
    short_term_power = short_term_data.get('total_power', 0)
    
    bars = ax2.bar(['Long-term', 'Short-term'], [long_term_power, short_term_power], 
                   color=['steelblue', 'orange'], alpha=0.7)
    ax2.set_title('Total Power Consumption', fontweight='bold')
    ax2.set_ylabel('Power (mW)')
    
    for bar, value in zip(bars, [long_term_power, short_term_power]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Rate achievement comparison
    ax3 = axes[0, 2]
    if 'rates_per_slice' in long_term_data and 'rates_per_slice' in short_term_data:
        long_rates = long_term_data['rates_per_slice']
        short_rates = short_term_data['rates_per_slice']
        
        bars1 = ax3.bar(x - width/2, long_rates, width, label='Long-term', color='lightblue', alpha=0.8)
        bars2 = ax3.bar(x + width/2, short_rates, width, label='Short-term', color='lightpink', alpha=0.8)
        
        ax3.set_title('Rate Achievement Comparison', fontweight='bold')
        ax3.set_xlabel('Slice Type')
        ax3.set_ylabel('Rate (Mbps)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(slices)
        ax3.legend()
    
    # Plots 4-6: Individual metrics for each solution
    # Long-term spectral efficiency
    ax4 = axes[1, 0]
    if 'spectral_eff' in long_term_data:
        ax4.bar(range(len(long_term_data['spectral_eff'])), long_term_data['spectral_eff'], 
                color='steelblue', alpha=0.7)
        ax4.set_title('Long-term Spectral Efficiency', fontweight='bold')
        ax4.set_xlabel('RB Index')
        ax4.set_ylabel('Efficiency (bps/Hz)')
    
    # Short-term spectral efficiency
    ax5 = axes[1, 1]
    if 'spectral_eff' in short_term_data:
        ax5.bar(range(len(short_term_data['spectral_eff'])), short_term_data['spectral_eff'], 
                color='coral', alpha=0.7)
        ax5.set_title('Short-term Spectral Efficiency', fontweight='bold')
        ax5.set_xlabel('RB Index')
        ax5.set_ylabel('Efficiency (bps/Hz)')
    
    # Efficiency difference
    ax6 = axes[1, 2]
    if 'spectral_eff' in long_term_data and 'spectral_eff' in short_term_data:
        diff = np.array(short_term_data['spectral_eff']) - np.array(long_term_data['spectral_eff'])
        colors = ['green' if d >= 0 else 'red' for d in diff]
        ax6.bar(range(len(diff)), diff, color=colors, alpha=0.7)
        ax6.set_title('Efficiency Difference (Short - Long)', fontweight='bold')
        ax6.set_xlabel('RB Index')
        ax6.set_ylabel('Efficiency Difference')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = f"comparison_frame{frame_num}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_solution_summary(solution_data, solution_type, frame_num, time_slot=None, 
                         save_path="./result", show_plot=True, save_plot=True):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create title
    if time_slot is not None:
        title = f'{solution_type.title()} Solution Summary - Frame {frame_num}, Time Slot {time_slot}'
    else:
        title = f'{solution_type.title()} Solution Summary - Frame {frame_num}'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Resource utilization pie chart
    ax1 = axes[0, 0]
    if 'resource_utilization' in solution_data:
        labels = list(solution_data['resource_utilization'].keys())
        sizes = list(solution_data['resource_utilization'].values())
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Resource Utilization Distribution')
    
    # Plot 2: Performance metrics bar chart
    ax2 = axes[0, 1]
    if 'performance_metrics' in solution_data:
        metrics = solution_data['performance_metrics']
        ax2.bar(metrics.keys(), metrics.values(), color='lightgreen', alpha=0.7)
        ax2.set_title('Performance Metrics')
        ax2.set_ylabel('Value')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Power vs Rate scatter
    ax3 = axes[1, 0]
    if 'power_rate_data' in solution_data:
        power_data = solution_data['power_rate_data']['power']
        rate_data = solution_data['power_rate_data']['rate']
        ax3.scatter(power_data, rate_data, alpha=0.6)
        ax3.set_xlabel('Power (mW)')
        ax3.set_ylabel('Rate (bps)')
        ax3.set_title('Power vs Rate Trade-off')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Constraints satisfaction
    ax4 = axes[1, 1]
    if 'constraints' in solution_data:
        constraints = solution_data['constraints']
        satisfied = sum(1 for v in constraints.values() if v)
        total = len(constraints)
        
        labels = ['Satisfied', 'Violated']
        sizes = [satisfied, total - satisfied]
        colors = ['green', 'red']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
        ax4.set_title(f'Constraints Status ({satisfied}/{total})')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = f"{solution_type}_summary_frame{frame_num}"
        if time_slot is not None:
            filename += f"_timeslot{time_slot}"
        filename += ".png"
        
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"{solution_type.title()} summary plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()