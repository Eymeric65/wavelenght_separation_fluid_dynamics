import os
import json
import hashlib
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def calculate_radial_profile(fft_shifted):
    """
    Calculates the 1D radial profile of a 2D FFT magnitude spectrum.

    Args:
        fft_shifted (np.ndarray): The 2D numpy array of the FFT, with the
                                  zero-frequency component shifted to the center.

    Returns:
        tuple: A tuple containing:
            - radial_bins (np.ndarray): The center of each radial bin (the x-axis).
            - radial_profile (np.ndarray): The mean magnitude for each bin (the y-axis).
    """
    # 1. Get the magnitude spectrum and the image dimensions
    magnitude_spectrum = np.abs(fft_shifted)
    height, width = magnitude_spectrum.shape
    center_y, center_x = height // 2, width // 2

    # 2. Create a grid of radial distances from the center
    y, x = np.indices((height, width))
    radius_grid = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # 3. Bin the radii into integer values
    # These integer radii will serve as indices for our bins
    radius_bins_int = radius_grid.astype(int)
    
    # 4. Calculate the sum of magnitudes for each radial bin
    # 'weights' uses the magnitude at each pixel
    # The length of 'tbin' will be max_radius + 1
    radial_sum = np.bincount(radius_bins_int.ravel(), weights=magnitude_spectrum.ravel())
    
    # 5. Calculate the number of pixels in each radial bin
    nr = np.bincount(radius_bins_int.ravel())
    
    # 6. Calculate the mean by dividing the sum by the count
    # Avoid division by zero for bins with no pixels
    radial_profile = np.divide(radial_sum, nr, out=np.zeros_like(radial_sum), where=nr!=0)
    
    # The x-axis for our plot is simply the bin number (i.e., the radius)
    radial_bins = np.arange(len(radial_profile))

    return radial_bins, radial_profile

def hanning_window_2d(image:np.ndarray) -> np.ndarray:
    """
    Apply a 2D Hanning window to the input image.

    Args:
        image (np.ndarray): 2D array representing the image.

    Returns:
        np.ndarray: The windowed image.
    """
    nx, ny = image.shape
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    window = np.outer(wx, wy)
    return image * window




def import_data(foldername):
    """
    Python conversion of the cluster_DMD_modes MATLAB script.
    
    This function loads Dynamic Mode Decomposition (DMD) results, processes them to
    extract features using FFT, and then clusters the modes based on their frequency
    and spatial correlation.

    Args:
        foldername (str): The directory where the binary data files are located.

    Returns:
        tuple: A tuple containing:
            - phi (np.ndarray): The complex-valued DMD modes.
            - freq (np.ndarray): The frequencies of the modes.
            - om_list (np.ndarray): The complex eigenvalues.
            - b_list (np.ndarray): The mode amplitudes.
    """
    # --- 1. Constants and File Path Construction ---
    nx = 513
    ny = 257
    freq_tol = 0.2e6
    corr_tol = 0.9

    # Define file paths
    omega_file = os.path.join(foldername, 'omega_dmd_ey.bin')
    phi_real_file = os.path.join(foldername, 'phi_real_dmd_ey.bin')
    phi_imag_file = os.path.join(foldername, 'phi_imag_dmd_ey.bin')
    b_file = os.path.join(foldername, 'b_dmd_ey.bin')

    # --- 2. File I/O - Reading Binary Data ---
    try:
        # Use np.fromfile to read binary data with 'float64' (double precision)
        phi_real = np.fromfile(phi_real_file, dtype=np.float64)
        phi_imag = np.fromfile(phi_imag_file, dtype=np.float64)
        omega = np.fromfile(omega_file, dtype=np.float64)
        b = np.fromfile(b_file, dtype=np.float64)
    except FileNotFoundError as e:
        print(f"Error: Binary file not found. {e}")
        print(f"Searched in folder: {foldername}")
        return None, None, None, None, None

    # --- 3. Data Reshaping and Initial Processing ---

    nmodes = len(omega) // 2  # Number of modes

    # Reshape arrays. 'F' order is crucial to match MATLAB's column-major order.
    phi_real = phi_real.reshape((nx, ny, nmodes), order='F')
    phi_imag = phi_imag.reshape((nx,ny, nmodes), order='F')

    phi = phi_real + 1j * phi_imag
    
    # Note: Python uses 0-based indexing.
    # MATLAB's omega(r+1:end) is omega[r:] in Python.
    # MATLAB's omega(1:r) is omega[:r] in Python.
    freq = np.abs(omega[nmodes:]) / (2 * np.pi)
    om_list = omega[:nmodes] + 1j * omega[nmodes:]
    b_list = b[:nmodes] + 1j * b[nmodes:]

    return phi, freq, om_list, b_list


if __name__ == '__main__':

    #Input 

    fmin = 0.1e6
    fmax = 8.0e6
    # First zone is inside the thruster and second one is outside
    first_zone_x = [0, 100]
    second_zone_x = [200, 500]

    eps_dbscan = 0.55

    cutoff_radius = 35  # Example cutoff radius in pixels
    feature_bin_count= 60  # Number of bins for feature extraction

    dbscan_min_samples = 5

    # Save input parameters to JSON
    input_params = {
        'fmin': fmin,
        'fmax': fmax,
        'first_zone_x': first_zone_x,
        'second_zone_x': second_zone_x,
        'eps_dbscan': eps_dbscan,
        'cutoff_radius': cutoff_radius,
        'feature_bin_count': feature_bin_count,
        'dbscan_min_samples': dbscan_min_samples
    }
    
    # Generate unique ID from hash of input parameters
    params_string = json.dumps(input_params, sort_keys=True)
    run_uid = hashlib.md5(params_string.encode()).hexdigest()[:8]
    
    # Add metadata to params
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_params['run_uid'] = run_uid
    input_params['timestamp'] = timestamp

    # Create results directory with UID
    result_dir = f'result_{run_uid}'
    os.makedirs(result_dir, exist_ok=True)
    print(f'\nSaving results to {result_dir}/')

    with open(os.path.join(result_dir, 'input_parameters.json'), 'w') as f:
        json.dump(input_params, f, indent=4)
    print(f'Run UID: {run_uid} (hash of input parameters)')
    print(f'Saved input parameters to {result_dir}/input_parameters.json')
    # 1 - Import data

    print("Creating dummy data for demonstration...")
    foldername = 'data'

    os.makedirs(foldername, exist_ok=True)

    phi,freq,om_list,b_list = import_data(foldername)

    print("Number of modes imported:", len(freq))

    # 2 - Filter based on frequency

    valid_idx = (freq >= fmin) & (freq <= fmax)

    phi = phi[:,:, valid_idx]
    freq = freq[valid_idx]
    om_list = om_list[valid_idx]
    b_list = b_list[valid_idx]

    print("Number of modes after frequency filtering:", len(freq))

    # 3 - divide phi into two zones

    phi_first_zone = np.real(phi[first_zone_x[0]:first_zone_x[1], :, :]).copy()
    phi_second_zone = np.real(phi[second_zone_x[0]:second_zone_x[1], :, :]).copy()

    # 4 - Apply Hanning window to the two zones for the first mode

    for iMode in range(phi_first_zone.shape[2]):
        phi_first_zone[:,:,iMode] = hanning_window_2d(phi_first_zone[:,:,iMode])
        phi_second_zone[:,:,iMode] = hanning_window_2d(phi_second_zone[:,:,iMode])

    # 5 - FFT of the two zones for the first mode

    fft_first_zone = np.fft.fft2(phi_first_zone,axes=(0,1))
    fft_second_zone = np.fft.fft2(phi_second_zone,axes=(0,1))

    # FFT shift to center the zero frequency component
    fft_first_zone = np.fft.fftshift(fft_first_zone)
    fft_second_zone = np.fft.fftshift(fft_second_zone)

    # 6 - takes the magnitude of the FFTs

    mag_fft_first_zone = np.abs(fft_first_zone)
    mag_fft_second_zone = np.abs(fft_second_zone)

    # 7 - Calculate radial profiles for modes

    radial_bins_first = []
    radial_profile_first = []
    radial_bins_second = []
    radial_profile_second = []



    bin_edges = np.linspace(0, cutoff_radius, feature_bin_count + 1)

    feature_vec_f = np.zeros((feature_bin_count, mag_fft_first_zone.shape[2]))
    feature_vec_s = np.zeros((feature_bin_count, mag_fft_first_zone.shape[2]))
    
    for iMode in range(mag_fft_first_zone.shape[2]):

        r_bins_f, r_prof_f = calculate_radial_profile(mag_fft_first_zone[:,:,iMode])
        r_bins_s, r_prof_s = calculate_radial_profile(mag_fft_second_zone[:,:,iMode])

        radial_bins_first.append(r_bins_f)
        radial_profile_first.append(r_prof_f)
        radial_bins_second.append(r_bins_s)
        radial_profile_second.append(r_prof_s)
    
        # Classical histogram without peak selection

        feature_vec_f[:,iMode], _ = np.histogram(r_bins_f, bins=bin_edges, weights=r_prof_f)
        feature_vec_s[:,iMode], _ = np.histogram(r_bins_s, bins=bin_edges, weights=r_prof_s)

        
    
    # Convert lists to numpy arrays, stacking along axis 1 (columns)
    radial_bins_first = np.column_stack(radial_bins_first)
    radial_profile_first = np.column_stack(radial_profile_first)
    radial_bins_second = np.column_stack(radial_bins_second)
    radial_profile_second = np.column_stack(radial_profile_second)

    # 8 - Clustering using DBSCAN

    # Normalize feature vectors
    feature_vec_f_norm = feature_vec_f / np.linalg.norm(feature_vec_f, axis=0, keepdims=True)
    feature_vec_s_norm = feature_vec_s / np.linalg.norm(feature_vec_s, axis=0, keepdims=True)   

    # Concatenate feature vectors from both zones
    combined_features = np.vstack((feature_vec_f_norm, feature_vec_s_norm)).T

    # # Concatenate feature vectors from both zones
    # combined_features = np.vstack((feature_vec_f, feature_vec_s)).T
    # combined_features = combined_features / np.linalg.norm(combined_features, axis=1, keepdims=True)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=dbscan_min_samples)
    cluster_labels = dbscan.fit_predict(combined_features)
    print("number of modes filtered",len(cluster_labels),"Cluster labels for each mode:", cluster_labels," Total clusters found:", len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))

    # 9 - Visualization



    # First, create separate figures for each cluster showing all modes in that cluster
    unique_labels = set(cluster_labels)
    for k in sorted(unique_labels):
        if k == -1:
            label_str = 'Noise'
            filename_label = 'noise'
        else:
            label_str = f'Cluster {k}'
            filename_label = f'cluster_{k}'
        
        # Get all mode indices for this cluster
        class_member_mask = (cluster_labels == k)
        mode_indices = np.where(class_member_mask)[0]
        
        if len(mode_indices) == 0:
            continue
        
        # Calculate grid size for subplots
        n_modes = len(mode_indices)
        n_cols = min(5, n_modes)  # Max 5 columns
        n_rows = int(np.ceil(n_modes / n_cols))
        
        # Create figure for this cluster
        fig_cluster = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
        fig_cluster.suptitle(f'{label_str} - Spatial Maps of DMD Modes', fontsize=16, fontweight='bold')
        
        for idx, mode_idx in enumerate(mode_indices):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Plot the real part of phi for this mode
            mode_data = np.real(phi[:, :, mode_idx])
            im = ax.imshow(mode_data.T, cmap='seismic', aspect='auto')
            ax.set_title(f'Mode {mode_idx}\nf={freq[mode_idx]/1e6:.2f} MHz', fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Remove empty subplots if any
        # for idx in range(n_modes, n_rows * n_cols):
        #     fig_cluster.delaxes(fig_cluster.axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'{result_dir}/{filename_label}_spatial_maps.png', dpi=300, bbox_inches='tight')
        plt.close(fig_cluster)
        print(f'Saved: {result_dir}/{filename_label}_spatial_maps.png')
    
    # Create figure with multiple subplots for comprehensive cluster visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Scatter plot: Frequency vs Mode Index colored by cluster
    ax1 = plt.subplot(2, 3, 1)
    unique_labels = set(cluster_labels)
    colors_map = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors_map):
        if k == -1:
            # Black for noise points
            col = [0, 0, 0, 1]
        
        class_member_mask = (cluster_labels == k)
        xy = np.column_stack([np.arange(len(cluster_labels))[class_member_mask], 
                               freq[class_member_mask]])
        
        label_str = f'Cluster {k}' if k != -1 else 'Noise'
        ax1.scatter(xy[:, 0], xy[:, 1]/1e6, c=[col], s=100, 
                   label=label_str, edgecolors='k', linewidths=0.5)
    
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Frequency (MHz)')
    ax1.set_title('Clusters by Frequency')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of cluster sizes
    ax2 = plt.subplot(2, 3, 2)
    cluster_counts = {}
    for label in unique_labels:
        count = np.sum(cluster_labels == label)
        label_str = f'C{label}' if label != -1 else 'Noise'
        cluster_counts[label_str] = count
    
    bars = ax2.bar(cluster_counts.keys(), cluster_counts.values(), 
                   color=colors_map, edgecolor='k')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Modes')
    ax2.set_title('Cluster Size Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 3. Feature space visualization (PCA projection)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(combined_features)
    
    ax3 = plt.subplot(2, 3, 3)
    for k, col in zip(unique_labels, colors_map):
        if k == -1:
            col = [0, 0, 0, 1]
        
        class_member_mask = (cluster_labels == k)
        xy = features_2d[class_member_mask]
        
        label_str = f'Cluster {k}' if k != -1 else 'Noise'
        ax3.scatter(xy[:, 0], xy[:, 1], c=[col], s=100, 
                   label=label_str, edgecolors='k', linewidths=0.5)
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax3.set_title('Feature Space (PCA Projection)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap of feature vectors grouped by cluster
    ax4 = plt.subplot(2, 3, 4)
    # Sort modes by cluster label
    sort_idx = np.argsort(cluster_labels)
    sorted_features = combined_features[sort_idx, :]
    sorted_labels = cluster_labels[sort_idx]
    
    im = ax4.imshow(sorted_features.T, aspect='auto', cmap='viridis')
    ax4.set_xlabel('Mode Index (sorted by cluster)')
    ax4.set_ylabel('Feature Dimension')
    ax4.set_title('Feature Vectors Grouped by Cluster')
    
    # Add cluster boundaries
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
    for boundary in boundaries:
        ax4.axvline(boundary, color='red', linewidth=2, linestyle='--')
    
    plt.colorbar(im, ax=ax4, label='Feature Value')
    
    # 5. Average radial profiles per cluster (first zone)
    ax5 = plt.subplot(2, 3, 5)
    for k, col in zip(unique_labels, colors_map):
        if k == -1:
            continue  # Skip noise
        
        class_member_mask = (cluster_labels == k)
        avg_profile = np.mean(radial_profile_first[:, class_member_mask], axis=1)
        ax5.plot(radial_bins_first[:, 0], avg_profile, 
                color=col, linewidth=2, label=f'Cluster {k}')
    
    ax5.set_xlabel('Radial Distance (pixels)')
    ax5.set_ylabel('Mean FFT Magnitude')
    ax5.set_title('Average Radial Profiles - First Zone')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Average radial profiles per cluster (second zone)
    ax6 = plt.subplot(2, 3, 6)
    for k, col in zip(unique_labels, colors_map):
        if k == -1:
            continue  # Skip noise
        
        class_member_mask = (cluster_labels == k)
        avg_profile = np.mean(radial_profile_second[:, class_member_mask], axis=1)
        ax6.plot(radial_bins_second[:, 0], avg_profile, 
                color=col, linewidth=2, label=f'Cluster {k}')
    
    ax6.set_xlabel('Radial Distance (pixels)')
    ax6.set_ylabel('Mean FFT Magnitude')
    ax6.set_title('Average Radial Profiles - Second Zone')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/cluster_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {result_dir}/cluster_analysis_summary.png')

    print(f'\nAll figures saved to {result_dir}/ directory!')
    print(f'Input parameters saved to result/input_parameters.json')
    exit()

    # # Plotting DEBUG

    # # Plot the first mode image
    # plt.imshow(phi_first_zone[:,:,0].T, cmap='seismic')
    # plt.title('First Mode Spatial Map')
    # plt.show()

    # plt.imshow(phi_second_zone[:,:,0].T, cmap='seismic')
    # plt.title('Second Mode Spatial Map')
    # plt.show()


    # plt.imshow(mag_fft_first_zone[:,:,0].T, cmap='jet')
    # plt.title('First Zone - Mode 0')
    # plt.show()

    # plt.imshow(mag_fft_second_zone[:,:,0].T, cmap='jet')
    # plt.title('Second Zone - Mode 0')
    # plt.show()

    # plt.imshow( feature_vec_f, aspect='auto', cmap='viridis')
    # plt.title('Feature Vector Heatmap - First Zone')
    # plt.xlabel('Mode Index')
    # plt.ylabel('Feature Bin Index')
    # plt.colorbar(label='Amplitude-Weighted Count')
    # plt.show()

    # plt.imshow( feature_vec_s, aspect='auto', cmap='viridis')
    # plt.title('Feature Vector Heatmap - Second Zone')
    # plt.xlabel('Mode Index')
    # plt.ylabel('Feature Bin Index')
    # plt.colorbar(label='Amplitude-Weighted Count')
    # plt.show()

    # # Plot radial profiles
    # plt.figure()
    # for iMode in range(radial_profile_first.shape[1]):
    #     plt.plot(radial_bins_first[:,iMode], radial_profile_first[:,iMode], label=f'First Zone {iMode}')
    #     plt.plot(radial_bins_second[:,iMode], radial_profile_second[:,iMode], label=f'Second Zone {iMode}')

    # plt.xlabel('Radial Distance (pixels)')
    # plt.ylabel('Mean FFT Magnitude')
    # plt.title(f'Radial Profile - Mode {iMode}')
    # plt.legend()
    # plt.show()
